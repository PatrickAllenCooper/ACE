# Bounded angle-offset calibration diagnostic

Post hoc development diagnostic motivated by the failed fixed physics basis. Reuse only white_64 and the previously committed blocked split; do not claim fresh confirmation. Other conditions remain unscored.

Fit y = intercept + amplitude*cos(theta1 + offset1)^2*cos(theta2-theta1 + offset2)^2. Angle scale stays fixed; offsets absorb apparatus reference conventions. This is one candidate physical approximation, not a complete calibrated optical model. No sensor-angle features.

Use scipy least_squares on training residuals normalized by training standard deviation. Offset bounds [-pi,pi], amplitude nonnegative, intercept unbounded. Sixteen fixed starts from offsets {-pi/2,-pi/4,0,pi/4} squared, intercept training min, amplitude training range. Each fit at most 500 evaluations. Select minimum training squared error only. Record convergence, fitted parameters, training/test normalized errors. No changing model or starts after test access. Stop after this diagnostic and reassess whether mechanism mismatch remains; no GPU escalation. Two-minute local CPU cap.
