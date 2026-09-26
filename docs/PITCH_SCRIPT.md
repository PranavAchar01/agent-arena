# Handoff: pitch script (Agent Arena, Vultr x Cerebral Valley, track partner NetBird)

About 2 minutes 45 seconds spoken. Lines marked **[IF BUILT]** are only true once that piece runs at the event;
the fallback wording under each one is true today. Everything else is true now and measured (docs/STATS.md).

---

## 1. The problem (0:00 to 0:35)

Every robot demo you have seen was trained on data somebody collected by hand. A person puppeteers the arm, over
and over, for one task. Then you want a second task, and you do it all again.

That is the bottleneck in robotics right now. Not the models. The data.

Meanwhile, the open web is full of people doing these exact tasks with their hands. Nobody trains robots on it, for
two reasons.

One: it is dangerous. An agent that goes out to the open web is an agent that follows redirects, downloads whatever
it is handed, and reads text that is trying to hijack it.

Two: it is the wrong body. A human hand is not a robot gripper.

## 2. Who it is for (0:35 to 0:50)

Handoff is for anyone who has a robot arm and no data team. A small robotics startup testing a new task on Monday.
A lab with a low-cost open-source arm like the SO-101. A classroom. You type what you want the robot to do, and you
get back a trained policy, tested, ready to deploy.

## 3. Live demo (0:50 to 1:50)

*(Type: "put the block in the bowl". Press Train.)*

Every prompt becomes its own robot, and every robot gets its own sandbox.

**[IF BUILT]** That box is a throwaway Vultr VM that did not exist two seconds ago.
*Fallback: That box is a throwaway sandbox: in production, one Vultr VM per robot.*

*(Point at the terminal.)* This is the scraping agent working inside it. No secrets on the machine. CPU, memory and
time are capped. We plant hostile pages on purpose: a redirect loop, an endless download, a page that tells the
agent to send us its API keys, a fake video. Watch them get blocked. Nothing from that sandbox reaches our app
except verified clips and numbers.

*(Point at the verify lines.)* A vision model checks every clip: is this a real person, a real hand, the right
motion?

**[IF BUILT]** That model runs on Vultr Serverless Inference.
*Fallback: The model sits behind an OpenAI-compatible endpoint; pointing it at Vultr Serverless Inference is one line of config.*

*(Point at the tuning terminal.)* Now it switches to training. MediaPipe tracks the human hand, we map that motion
onto the SO-101, and every single episode has to physically work in a physics engine, or it is thrown out. Then we
train the robot's policy on what survives, on a plain CPU, in about 21 seconds.

*(Type two more prompts: "stack the block on another block", "add one more block to make a tower".)*

Three prompts, three robots, three sandboxes, running side by side.

*(When the boxes turn green.)* Each one was tested on twenty layouts it has never seen: 19, 18 and 18 out of 20.

## 4. This is real physics, and a real trained model (1:50 to 2:15)

I want to be very clear about what you are watching. This is not an animation.

It is MuJoCo, the physics engine Google DeepMind open-sourced, and the same engine their team used to train soccer
robots that then played on real hardware with no extra training. Gravity, friction and contact are all simulated;
the grip is what holds the block up. The arm is the real SO-101 model: its CAD meshes, its joint limits, its servo
gains.

And the policy is a real trained network, 166 thousand parameters, that outputs the same six joint commands the
physical arm takes. Next, we run it on a real SO-101.

## 5. Why Vultr, and why NetBird (2:15 to 2:35)

Containment is the whole product, and that is what Vultr gives us.

**[IF BUILT]** One prompt is one Vultr VM. A hundred robots is a hundred VMs that live for ten minutes and then
disappear. The models run on Vultr Serverless Inference, and the whole pipeline runs on CPU compute. No GPU anywhere.
*Fallback: Today each sandbox is a local container behind the same interface; the Vultr backend creates a VM per
robot and deletes it when the job ends.*

**[IF BUILT]** And NetBird is the only door out of each sandbox. Every VM joins a private NetBird network with one
rule: it may stream its log to our app, and nothing else. No public ports. When the VM dies, its peer and its key die
with it.
*Fallback: Next, each VM joins a private NetBird network whose only rule is "stream your log to the app".*

## 6. Close (2:35 to 2:45)

We planted 85 hostile pages across 17 runs. Zero got through.

Handoff. One prompt, one robot, one sandbox. The open web is full of people showing robots how to do things. We made
it safe to learn from.

---

## 60-second voiceover for the film (optional, matches film/handoff-demo.mp4)

| time | line |
|---|---|
| 0:00 | Robots learn from demonstrations, and today somebody has to record every one by hand. |
| 0:04 | Handoff learns from people on the open web instead. You just type what you want. |
| 0:08 | Each robot gets its own throwaway sandbox. The scraper works inside it; hostile pages get blocked, and a vision model checks every clip. |
| 0:20 | Then the human hand motion is mapped onto an SO-101, checked in a physics engine, and a policy is trained on a laptop CPU. |
| 0:30 | Three prompts, three robots, each tested on twenty layouts it has never seen. This part is sped up about thirty times. |
| 0:43 | And this is not an animation. It is MuJoCo, the physics engine DeepMind used to train robots that then worked on real hardware. |
| 0:51 | Handoff. One prompt, one robot, one sandbox. |

---

## Q&A prep (short, true answers)

**Does it work on a real robot?**
Not tested on hardware yet. It is real physics on the real SO-101 model, and the policy outputs the same joint
commands the physical arm takes. That transfer test is our next step; I have an SO-101 arm assembled.

**Did you fine-tune a foundation model?**
We tested it. A pretrained ResNet-18 from Hugging Face looking at camera frames scored 11 out of 50 frozen and 5 out
of 50 fine-tuned on CPU, against 40 out of 50 for our small policy. With a day and no GPU, the small policy wins, so
that is what we ship. Bigger vision-language-action models are too slow to tune on CPU.

**Is the data really from the web?**
The scraper searches openly licensed archives with licence filters: CC0, CC BY, CC BY-SA, public domain. For these
three robots the verified clips came from HO-Cap, an open CC BY 4.0 dataset of real people at a table, read straight
out of its archive inside the sandbox. We never use YouTube, because its terms forbid downloading.

**What stops a malicious page?**
It never touches the app. It runs in a throwaway sandbox with no secrets, a read-only filesystem, no privileges,
CPU, memory and time caps, and a hard kill. The model only sees page titles as data and can only answer with line
numbers. What comes back is checked against a strict format before we read it.

**Why human video instead of generating demos in simulation?**
Because people already know how to do the task. Their motion gives the timing and the path, the physics engine
checks it works for the robot, and the robot never needs a human puppeteer.

**How long does one robot take?**
About 10 to 11 minutes end to end on a laptop, mostly downloading and reading video. Tuning is about 21 seconds.
The film is sped up about 30 times.
