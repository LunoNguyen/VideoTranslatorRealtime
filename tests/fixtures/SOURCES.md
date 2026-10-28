# Fixture sources

- `en_hello.wav`: generated locally with Windows SAPI voice "Microsoft Zira Desktop" (en-US), text "Hello, how are you today?", 16 kHz mono 16-bit.
- `ja_konnichiwa.wav`: not generated; no Japanese SAPI voice is installed. The gpu test skips if missing. Add a 16 kHz mono recording of "こんにちは" manually to enable it.
- `spk_david_1.wav`, `spk_david_2.wav`: Windows SAPI voice "Microsoft David Desktop" (en-US), texts "The weather is lovely this afternoon, let's go for a walk." and "I think we should finish the report before dinner tonight."; 16 kHz mono 16-bit.
- `spk_zira_1.wav`: Windows SAPI voice "Microsoft Zira Desktop" (en-US), same text as `spk_david_1.wav`; 16 kHz mono 16-bit. Used by the speaker-embedder voice-separation test.
