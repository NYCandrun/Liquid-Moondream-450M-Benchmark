# EDGE RUSH

You are the AI chip.

A tiny browser game about how modern AI runs right on your phone, car, and glasses instead of in a faraway data center. No experience needed. It teaches one idea at a time.

It is inspired by Liquid AI, a company building AI that runs directly on devices. The game turns their core ideas into something you can feel by playing.

## How to play

Open `index.html` in Chrome. That is it. No install, no sign-up, no internet required.

Requests stream in at the bottom of the screen. Each one has tags (short or LONG, simple or COMPLEX). Your job is to send each request to the right place before its timer runs out, while keeping your battery, memory (RAM), and accuracy healthy.

Use the number keys, or click the buttons:

| Key | Block | What it is good at |
|-----|-------|-------------------|
| 1 | FAST | On-device, cheap, instant, but only for short and simple requests |
| 2 | ATTENTION | Handles anything hard, but its memory keeps growing and burns battery |
| 3 | NANO | A tiny specialist that is perfect at one job and useless at the rest |
| 4 | CLOUD | Always gives a great answer, but it is slow and dies when you lose signal |

Tap the question mark in the corner anytime to re-read the current lesson. It pauses the game. Your progress and best score are saved automatically on your device.

## The levels

There are six levels. Each one is a short playable round that starts with a quick explanation and adds exactly one new idea on top of the last. New buttons and meters only appear when their level introduces them, so the screen never gets ahead of you.

**Level 1: Where does AI run?**
You are a phone. Only two choices exist: run a request on the device (FAST) or send it to the CLOUD. You learn that on-device is instant and private, while the cloud is smart but slow and needs internet. Clear it by serving 6 requests.

**Level 2: Hard requests, and attention remembers**
Now requests can be LONG or COMPLEX, and the FAST block gets those wrong. A new ATTENTION block can handle anything, but it introduces the RAM meter. Attention has to remember everything it has read (the "KV-cache"), and the more it holds, the more each new call costs. Overfill the memory and you lose.

**Level 3: Power is not free**
You are a laptop now, and the BATTERY meter appears. Every block costs energy: FAST is almost nothing, ATTENTION and CLOUD cost a lot. Run out of battery and the device dies. The lesson is to pick the cheapest block that still gets the answer right.

**Level 4: You are the AI in a car**
The car drives into tunnels and loses signal. When that happens the CLOUD button goes dark and stops working. Your on-device blocks keep going no matter what. This is why a safety feature in a car cannot depend on the cloud.

**Level 5: Specialists**
One request type becomes a loaded specialist, and a NANO block appears. For that one type, NANO is the best pick of all: tiny, cheap, and perfect. For everything else it fails. You learn that a swarm of small specialists can beat one big do-everything model.

**Level 6: Design your chip**
The final level. You choose how to build your model (mostly FAST, balanced, or mostly ATTENTION), then survive a fast round with everything turned on. At the end the game shows how you routed every request and tells you which kind of AI you built. A balanced hybrid wins, which is exactly how Liquid builds its real models.

## What you will learn

By the end you will understand, just from playing: where AI runs (your device versus the cloud), why attention is powerful but its memory keeps growing, how to spend limited power wisely, why on-device AI keeps working with no signal, how tiny specialist models can beat one giant generalist, and why a balanced mix of blocks wins.

## The big takeaway

The smartest move is usually not to send everything to a giant brain in the cloud. Do most of the work on the device with cheap, fast blocks, save the expensive attention for the genuinely hard cases, and use small specialists where they fit. That is the future this game is about: millions of small, efficient AI models running everywhere, instead of one huge one far away.

## Tech

A single `index.html` file. Plain HTML, CSS, and JavaScript. No frameworks, no build step, no backend, no login. Progress is stored locally in your browser.
