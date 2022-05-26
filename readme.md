# playback-rs
This is a very _very_ simple library to handle playing audio files using [Symphonia](https://docs.rs/symphonia/) and [cpal](https://docs.rs/symphonia/).

## Run the Example
Symphonia is very slow when running in debug mode, so it is recommended to run the example in release mode:
```sh
cargo run --release --example=aplay -- song1.mp3 song2.flac song3.ogg
```
