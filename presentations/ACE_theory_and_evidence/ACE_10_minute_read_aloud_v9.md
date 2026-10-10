# Active Causal Experimentalism

Every experiment counts.

That is the idea behind ACE, Active Causal Experimentalism. Imagine developing a compressor. Each test takes equipment time, energy, and engineering attention. You can change a setting, measure the response, and learn something. The challenge is choosing the intervention that teaches you the most.

ACE uses a model of how changes travel through a system to make that choice. It connects the mechanics of the system with the decisions we make about testing it. I will show you that process, explain where foundation models fit, and finish with recorded results comparing ACE with random interventions.

---

Consider a simplified compressor test rig. We choose a drive command, which changes shaft speed, which changes pressure rise. The arrows show the direction of influence.

A structural causal model, or SCM, describes each of those relationships separately. One function predicts speed from the drive command. Another predicts pressure rise from speed. We connect those functions to predict what the whole system will do.

For our teaching example, all values are deviations around a reference condition. Speed equals two and a half times the drive command, and pressure rise equals three times speed. These simple equations illustrate the process. They are not a calibrated model of this compressor.

---

We can now distinguish two interventions.

First, change the drive command and let the system respond. Speed changes naturally, and pressure responds to speed. One recorded intervention gives us information about both relationships.

Second, use a test-rig controller to hold shaft speed at a chosen value. During that experiment, speed no longer follows the ordinary drive-command mechanism. That is why its incoming arrow disappears. Pressure still responds to speed, so we can learn about the downstream relationship.

ACE keeps that distinction when learning. A value we imposed cannot teach us how that variable would have behaved naturally. This is how the causal graph determines which observations update which mechanisms.

---

The same idea changes how we represent a larger system.

A local table records the output of one mechanism for combinations of its direct inputs. Here, A and B determine M. With five settings for each input, that table has twenty-five rows. We can connect several local mechanisms through the graph.

A joint table instead records the final output for every combination of all the system's inputs. With ten inputs, each having five settings, that means nearly ten million rows.

In this discrete example, nine local tables contain just two hundred twenty-five entries. ACE's learners fit functions rather than literal tables, but the organizing principle is the same: learn the parts and connect them.

---

Here is how large the representation difference becomes.

The exhaustive joint grid contains 9.77 million configurations. Sampling that grid randomly also revisits configurations. Reaching ninety-five percent expected coverage takes about 29.26 million random draws.

The local representation contains two hundred twenty-five entries, assuming we know the graph, observe the intermediate variables, and can access the required settings.

That explains the appeal of causal structure. We can reuse a local relationship across many combinations elsewhere in the system. These numbers describe a discrete coverage example. A flexible associative model can also generalize without filling every row. We will look at ACE's measured advantage over random intervention selection shortly.

---

This is the intervention loop. The SCM predicts what could happen. The selector compares permitted interventions. The environment receives the chosen intervention and returns measurements. Those measurements update the mechanisms left natural, and the loop repeats while budget remains.

Above it is the separate LLM policy path implemented in ACE. It starts from a pretrained language model. Supervised training teaches intervention commands from examples informed by the graph and current prediction errors. Candidate evaluation then produces preferred and less-preferred commands. Direct preference optimization, or DPO, encourages the LLM to favor the preferred command relative to a reference model.

That is the training objective. Its effectiveness must be tested on new systems with all queries counted. The performance curves in this presentation come from direct SCM scoring, called PEV, rather than the LLM policy. The proposed connection lets the LLM suggest candidates to an SCM scorer. The round nodes separate proposing, selecting, and executing an intervention.

---

Let us make one cycle completely explicit.

ACE starts with several possible slopes for each mechanism. For shaft speed, those slopes are one, two, and three. For pressure rise, they are two, three, and four.

The average model therefore predicts speed as twice the drive command, and pressure rise as three times speed. The members disagree, so the model also has uncertainty.

There are six legal interventions: three drive-command settings and three shaft-speed settings. We give ACE a budget of one response and a fixed rule for breaking tied scores. Negative numbers mean below-reference settings, not negative physical revolutions per minute.

---

The message here is simple: one intervention can inform two mechanisms.

ACE scores the six permitted interventions before spending its response budget. Changing the drive command to either end of its range receives a score of about point eight five. Holding shaft speed at either end receives about point four one. The zero settings score zero in this simplified example.

Changing the drive command can teach us about both speed and pressure, which gives it the higher uncertainty-reduction score. The two highest scores tie, so the declared ordering selects a drive command of minus one. The model predicts a speed deviation of minus two and a pressure-rise deviation of minus six.

---

One measured response now improves both predictions.

The environment returns drive command minus one, shaft speed minus two and a half, and pressure rise minus seven and a half. These measurements differ from the forecasts, giving the learner a correction signal.

The speed mechanism learns from the measured drive command and speed. The pressure mechanism learns from measured speed and pressure. Both remained natural, so both can use this intervention response.

After one illustrative gradient update, the forecasts move to minus two point two five for speed and minus six point seven five for pressure rise. Both move toward the observed values. The response budget is now exhausted.

---

The recorded experiments use larger synthetic systems. Here is an actual graph from one of those experiments, containing thirty linked mechanisms.

The highlighted intervention changes X seven. The highlighted downstream paths show where that change can propagate. This structure helps ACE identify which mechanisms an experiment might inform.

The first selected intervention in this recorded system set X seven to about three point eight six and purchased fifty responses. The learner then updated using eligible observations.

We have moved from the compressor teaching example to a recorded synthetic benchmark. The central operation stays the same: use the graph and current uncertainty to choose the next intervention.

---

Across twenty synthetic systems, ACE produced sixty-five percent lower final mean mechanism prediction error than random intervention selection. It finished ahead on nineteen of the twenty systems.

Both methods received the same budget of two thousand responses per system. Both used the same SCM learning machinery. The difference was how they selected interventions.

ACE's final mean error was approximately point zero one five four, compared with point zero four three nine for random selection. The charts show the recorded learning curves, with a closer view on the right.

Within this setting, choosing interventions through the causal model improved what the learner could predict from the same response budget.

---

The final chart puts the two measured policies alongside the SCM-free grid illustration.

At the same prediction-error target, ACE first crossed after five hundred intervention responses, compared with fourteen hundred for the random intervention policy. That is ten versus twenty-eight batches, or sixty-four percent fewer. The random policy samples permitted targets and values randomly, then learns with the same SCM machinery.

The third bar is much larger: 29.26 million queries. It describes random sampling of the discrete joint grid until ninety-five percent expected coverage. That is a different task and a calculated example. It is not a measured third policy at the same error target, and we cannot interpret its ratio to ACE as an observed speedup.

The measured comparison shows the value of choosing informative interventions. The grid example shows why structure can make a problem easier to represent.

Active Causal Experimentalism. Every experiment counts.
