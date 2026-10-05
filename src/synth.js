/* Demo synth engine for Project Pythagoras.
 *
 * Plays the demo track's real measured telemetry — Dm / Bb / F / C at
 * 92.4 BPM — through the Web Audio API so the dashboard (waveform playhead,
 * segment + chord highlighting, lyric sync) is fully live without bundled
 * audio files. Real stem audio replaces this once the ingestion pipeline
 * lands.
 */

const CHORD_VOICINGS = {
  Dm: { pad: [146.83, 174.61, 220.0, 293.66], bass: 73.42 },
  Bb: { pad: [116.54, 146.83, 174.61, 233.08], bass: 58.27 },
  F: { pad: [87.31, 130.81, 174.61, 220.0], bass: 87.31 },
  C: { pad: [130.81, 164.81, 196.0, 261.63], bass: 65.41 },
};

export class SynthEngine {
  constructor({ bpm, chords, duration }) {
    this.bpm = bpm;
    this.chords = chords.length ? chords : ["Dm"];
    this.duration = duration;
    this.sixteenth = 60 / bpm / 4;
    this.barLen = this.sixteenth * 16;
    this.ctx = null;
    this.playing = false;
    this.deck = "A"; // A = full mix, B = stripped (pad + bass)
    this._baseTime = 0; // seconds into the track when the current run started
    this._startCtx = 0; // ctx.currentTime at run start
    this._step = 0;
    this._nextTime = 0;
    this._timer = null;
    this._noiseBuf = null;
    this.onended = null;
    this.onchord = null;
  }

  get totalSteps() {
    return Math.ceil(this.duration / this.sixteenth);
  }

  _ensureCtx() {
    if (this.ctx) return;
    const AC = window.AudioContext || window.webkitAudioContext;
    this.ctx = new AC();
    this.master = this.ctx.createGain();
    this.master.gain.value = 0.75;
    this.master.connect(this.ctx.destination);
    this.deckAGain = this.ctx.createGain();
    this.deckBGain = this.ctx.createGain();
    // Both decks share the same scheduled notes; the toggle crossfades
    // which submix is audible. Deck B drops the percussion bus.
    this.percBus = this.ctx.createGain();
    this.percBus.connect(this.deckAGain);
    this.deckAGain.connect(this.master);
    this.deckBGain.connect(this.master);
    this.setDeck(this.deck, true);
    const len = this.ctx.sampleRate * 0.5;
    this._noiseBuf = this.ctx.createBuffer(1, len, this.ctx.sampleRate);
    const data = this._noiseBuf.getChannelData(0);
    for (let i = 0; i < len; i++) data[i] = Math.random() * 2 - 1;
  }

  setDeck(deck, silent = false) {
    this.deck = deck;
    if (!this.ctx || silent) {
      if (this.ctx) this._applyDeck();
      return;
    }
    this._applyDeck();
  }

  _applyDeck() {
    const t = this.ctx.currentTime;
    const aFull = this.deck === "A";
    this.deckAGain.gain.setTargetAtTime(aFull ? 1 : 0, t, 0.05);
    this.deckBGain.gain.setTargetAtTime(aFull ? 0 : 1, t, 0.05);
  }

  play() {
    this._ensureCtx();
    if (this.ctx.state === "suspended") this.ctx.resume();
    if (this.playing) return;
    // Restart from the top if we were sitting at the end.
    if (this._baseTime >= this.duration - 0.05) this._baseTime = 0;
    this.playing = true;
    this._startCtx = this.ctx.currentTime + 0.06;
    this._step = Math.floor(this._baseTime / this.sixteenth);
    this._nextTime = this._startCtx;
    this._timer = setInterval(() => this._schedule(), 25);
  }

  pause() {
    if (!this.playing) return;
    this._baseTime = this.getTime();
    this.playing = false;
    clearInterval(this._timer);
    this._timer = null;
  }

  seek(seconds) {
    const t = Math.max(0, Math.min(this.duration, seconds));
    const wasPlaying = this.playing;
    if (wasPlaying) this.pause();
    this._baseTime = t;
    if (wasPlaying) this.play();
  }

  getTime() {
    if (!this.ctx) return this._baseTime;
    if (!this.playing) return this._baseTime;
    return Math.min(this.duration, this._baseTime + (this.ctx.currentTime - this._startCtx));
  }

  _schedule() {
    if (!this.playing) return;
    while (this._nextTime < this.ctx.currentTime + 0.18) {
      if (this._step >= this.totalSteps) {
        // Reached the end of the track.
        const over = this._nextTime - this.ctx.currentTime;
        setTimeout(() => {
          this.pause();
          this._baseTime = this.duration;
          if (this.onended) this.onended();
        }, Math.max(0, over * 1000));
        this._nextTime = Infinity;
        return;
      }
      this._playStep(this._step, this._nextTime);
      this._nextTime += this.sixteenth;
      this._step++;
    }
    // Keep the UI's chord display in sync.
    const bar = Math.floor(this.getTime() / this.barLen);
    const chord = this.chords[((bar % this.chords.length) + this.chords.length) % this.chords.length];
    if (chord !== this._lastChord) {
      this._lastChord = chord;
      if (this.onchord) this.onchord(chord, bar);
    }
  }

  _playStep(step, time) {
    const stepInBar = step % 16;
    const bar = Math.floor(step / 16);
    const chordName = this.chords[((bar % this.chords.length) + this.chords.length) % this.chords.length];
    const voice = CHORD_VOICINGS[chordName] || CHORD_VOICINGS.Dm;

    // Pad: re-articulate each bar + a softer pulse halfway through.
    if (stepInBar === 0 || stepInBar === 8) {
      const gain = stepInBar === 0 ? 0.16 : 0.09;
      for (const f of voice.pad) this._padNote(f, time, this.barLen * 0.55, gain, this.deckAGain, this.deckBGain);
    }
    // Bass: syncopated roots.
    if (stepInBar === 0 || stepInBar === 7 || stepInBar === 10) {
      this._bassNote(voice.bass, time, 0.5, 0.22);
    }
    // Hats on offbeats (deck A / full mix only).
    if (stepInBar % 4 === 2) {
      this._hat(time, stepInBar === 2 ? 0.05 : 0.035);
    }
    // Sparse sparkle arp, top octave (deck A only).
    if (stepInBar % 4 === 0) {
      const f = voice.pad[(step / 4) % voice.pad.length] * 2;
      this._pluck(f, time, 0.5, 0.05);
    }
  }

  _padNote(freq, time, dur, gain, ...buses) {
    for (const bus of buses) {
      const osc = this.ctx.createOscillator();
      osc.type = "sawtooth";
      osc.frequency.value = freq;
      osc.detune.value = (Math.random() - 0.5) * 8;
      const filter = this.ctx.createBiquadFilter();
      filter.type = "lowpass";
      filter.frequency.value = 900;
      filter.Q.value = 0.6;
      const g = this.ctx.createGain();
      g.gain.setValueAtTime(0, time);
      g.gain.linearRampToValueAtTime(gain, time + 0.6);
      g.gain.setValueAtTime(gain, time + dur - 0.4);
      g.gain.linearRampToValueAtTime(0, time + dur);
      osc.connect(filter).connect(g).connect(bus);
      osc.start(time);
      osc.stop(time + dur + 0.1);
    }
  }

  _bassNote(freq, time, dur, gain) {
    for (const bus of [this.deckAGain, this.deckBGain]) {
      const osc = this.ctx.createOscillator();
      osc.type = "sine";
      osc.frequency.value = freq;
      const g = this.ctx.createGain();
      g.gain.setValueAtTime(0, time);
      g.gain.linearRampToValueAtTime(gain, time + 0.02);
      g.gain.exponentialRampToValueAtTime(0.001, time + dur);
      osc.connect(g).connect(bus);
      osc.start(time);
      osc.stop(time + dur + 0.05);
    }
  }

  _hat(time, gain) {
    const src = this.ctx.createBufferSource();
    src.buffer = this._noiseBuf;
    const filter = this.ctx.createBiquadFilter();
    filter.type = "highpass";
    filter.frequency.value = 7000;
    const g = this.ctx.createGain();
    g.gain.setValueAtTime(gain, time);
    g.gain.exponentialRampToValueAtTime(0.001, time + 0.06);
    src.connect(filter).connect(g).connect(this.percBus);
    src.start(time);
    src.stop(time + 0.1);
  }

  _pluck(freq, time, dur, gain) {
    const osc = this.ctx.createOscillator();
    osc.type = "triangle";
    osc.frequency.value = freq;
    const g = this.ctx.createGain();
    g.gain.setValueAtTime(0, time);
    g.gain.linearRampToValueAtTime(gain, time + 0.01);
    g.gain.exponentialRampToValueAtTime(0.001, time + dur);
    osc.connect(g).connect(this.percBus);
    osc.start(time);
    osc.stop(time + dur + 0.05);
  }
}
