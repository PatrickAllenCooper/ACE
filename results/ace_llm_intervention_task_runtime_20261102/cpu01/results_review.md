# Target CPU integration results review

Independent reviewer: Mencius, separate bounded saved-artifact review on November 2, 2026. No model, test or prior acceptance was rerun.

Zero required corrections. The reviewer independently matched both copies of all 43 remote objects, totaling 4,908,699 bytes. Input pins, worker source, package versions, scheduler identity and actual resources reconcile with the frozen CPU packet. Job35207064 completed0:0.

The actual tiny Qwen checks record identical full and selected-position supervised losses and preference losses. All eight adapter gradient tensors have maximum absolute difference zero in each comparison. The fabricated tiny adapter reload also gives identical scores. Two generated commands are legal, bind their respective prompts and were not executed against an environment. The five task prompt lengths match the saved token profiles.

The two allocated CPU threads over163seconds give326allocatedCPU-seconds. Known cumulative allocation is2,290CPU-seconds and296GPU-slice-seconds. Conservative reservations remain30,000CPU-seconds and2,700GPU-seconds. The worker's process CPU, elapsed time, RSS and scheduler measurements are distinct and overlap. They are not a whole-cycle total.

This accepts tiny random Qwen CPU integration in the exact target runtime. It does not qualify full-size pretrained adapter restoration, GPU long-context execution, BF16 numerical equivalence, scientific training, intervention efficiency or engineering utility. The original studies and prior failed attempts remain unchanged.
