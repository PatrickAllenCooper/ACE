# Active Causal Experimentalism

Every experiment counts.

That is the idea behind ACE, Active Causal Experimentalism. Imagine developing a compressor. Each test takes equipment time, energy, and engineering attention. You can change a setting, measure the response, and learn something. The challenge is choosing the experiment that teaches you the most.

ACE uses a model of how changes travel through a system to make that choice. It connects the mechanics of the system with the decisions we make about testing it. I will show you that process, explain where foundation models fit, and finish with recorded results comparing ACE with random experiments.

---

Consider a simplified compressor test rig. We choose a drive command, which changes shaft speed, which changes pressure rise. The arrows show the direction of influence.

A structural causal model, or SCM, describes each of those relationships separately. One function predicts speed from the drive command. Another predicts pressure rise from speed. We connect those functions to predict what the whole system will do.

For our teaching example, all values are deviations around a reference condition. Speed equals two and a half times the drive command, and pressure rise equals three times speed. These simple equations illustrate the process. They are not a calibrated model of this compressor.

---

We can now distinguish two experiments.

First, change the drive command and let the system respond. Speed changes naturally, and pressure responds to speed. One recorded experiment gives us information about both relationships.

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

That explains the appeal of causal structure. We can reuse a local relationship across many combinations elsewhere in the system. These numbers describe a discrete coverage example. A flexible associative model can also generalize without filling every row. We will look at ACE's measured advantage over random experiment selection shortly.

---

This diagram brings the full mechanism together.

We supply a graph, eligible data, permitted controls, and a budget. In our proposed foundation-model extension, a pretrained model uses the data to suggest candidate predictors for individual mechanisms. Those candidates compete with numerical models and mechanisms we already have. Validation determines which candidates enter the SCM.

The SCM predicts responses to legal actions and represents uncertainty. The selector scores those actions and sends the chosen experiment to the environment. The environment returns measurements, which update the mechanisms left natural. Then the loop repeats.

For our compressor example, the actions are low, reference, or high drive command, or an independently controlled shaft-speed setting. The measured acquisition results later use the neural SCM loop. The foundation-model branch is the proposed extension.

---

Let us make one cycle completely explicit.

ACE starts with several possible slopes for each mechanism. For shaft speed, those slopes are one, two, and three. For pressure rise, they are two, three, and four.

The average model therefore predicts speed as twice the drive command, and pressure rise as three times speed. The members disagree, so the model also has uncertainty.

There are six legal actions: three drive-command settings and three shaft-speed settings. We give ACE a budget of one response and a fixed rule for breaking tied scores. Negative numbers mean below-reference settings, not negative physical revolutions per minute.

---

ACE now scores those six actions before spending the response.

Changing the drive command to either end of its range receives a score of about point eight five. Holding shaft speed at either end receives about point four one. The zero settings score zero in this simplified example.

The score estimates how much an observation could reduce uncertainty across the affected mechanisms. Changing the drive command is valuable because it can inform both downstream relationships.

The two highest scores tie. The declared ordering selects a drive command of minus one. Before running it, the model predicts a speed deviation of minus two and a pressure-rise deviation of minus six.

---

The environment returns the actual row: drive command minus one, shaft speed minus two and a half, and pressure rise minus seven and a half.

For the speed mechanism, ACE pairs the measured drive command with the observed speed. For the pressure mechanism, it pairs measured speed with observed pressure rise. Both mechanisms remained natural during this intervention.

After one illustrative gradient update, the forecasts move to minus two point two five for speed and minus six point seven five for pressure rise. They move toward the observed response.

One purchased experiment has improved two mechanisms. The budget is now zero, so the loop stops.

---

The recorded experiments use larger synthetic systems. Here is an actual graph from one of those experiments, containing thirty linked mechanisms.

The highlighted intervention changes X seven. The highlighted downstream paths show where that change can propagate. This structure helps ACE identify which mechanisms an experiment might inform.

The first selected action in this recorded system set X seven to about three point eight six and purchased fifty responses. The learner then updated using eligible observations.

We have moved from the compressor teaching example to a recorded synthetic benchmark. The central operation stays the same: use the graph and current uncertainty to choose the next experiment.

---

Across twenty synthetic systems, ACE produced sixty-five percent lower final mean mechanism prediction error than random experiment selection. It finished ahead on nineteen of the twenty systems.

Both methods received the same budget of two thousand responses per system. Both used the same SCM learning machinery. The difference was how they selected interventions.

ACE's final mean error was approximately point zero one five four, compared with point zero four three nine for random selection. The charts show the recorded learning curves, with a closer view on the right.

Within this setting, choosing experiments through the causal model improved what the learner could predict from the same response budget.

---

We can also ask when the recorded mean curves first reached the same error level. Using random selection's final mean error as the target, ACE first crossed it after ten intervention batches. Random selection first crossed after twenty-eight.

That is sixty-four percent fewer intervention batches: five hundred intervention responses instead of fourteen hundred. It is a retrospective comparison of the recorded curves. Both campaigns ran their full budgets.

The separate figure on the right returns to the joint-table illustration, with its 9.77 million entries. It explains representation size, while the measured bars show experiment selection.

The opportunity is to make every response more useful: organize learning around mechanisms, carry that knowledge through the graph, and choose experiments for the uncertainty they can resolve.

Active Causal Experimentalism. Every experiment counts.
