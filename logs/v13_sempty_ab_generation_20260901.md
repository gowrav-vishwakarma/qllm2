# v13_sempty A/B — generation comparison (2026-09-01)

Prefix-completion across the three tinystories-trained A/B arms, same prompts,
same seed, so the arms are directly comparable.

- seed=0, temp=0.8, top_k=50, top_p=0.9, repetition_penalty=1.2
- arms: complex-384 (100.36M, `ab_complex`), real-588 (101.89M param-matched,
  `ab_real_pm`), real-768 (161.97M width-matched, `ab_real`)
- checkpoints: `checkpoints_v13_sempty/ab_{complex,real_pm,real}/latest.pt`
- each arm is 1 epoch of tinystories (2172 steps, B8/T256, seed 42, bf16,
  fused_ce, grad-ckpt, lr 5e-5, warmup 100)

## Short completions (max_new_tokens=30) — grammar / recall / facts

### [char-recall] name + object continuity
PROMPT: Once upon a time, there was a little girl named Lily. She had a red ball. One day, Lily played with her
  complex-384 (100.4M)       | toys and her friends came to play with the dog. They were very happy. Suddenly, Timmy went to the park for a hug
  real-588 (101.9M, pm)      | toys for her friends. They were very happy to play with their mom's toy truck. "What happened?" Lily asked Lily for help.
  real-768 (162.0M)          | toys for a while, but she didn't know what to do. Lily went to the store and found a big box that made.

### [object-recall] color + object continuity
PROMPT: Tim had a blue car in his garage. He walked to the garage and drove the
  complex-384 (100.4M)       | dog on her hand. "Wow, you can't worry," said his mom. "I'm sorry I found a big mess." Tim
  real-588 (101.9M, pm)      | car on the ground. His mom said, "Can we play with me?" Timmy asked. His mom smiled and said, "Yes
  real-768 (162.0M)          | car on the ground. "Wow, you can't go inside!" Tim said. "I want to open it," he replied. "

### [fact-number] arithmetic / fact consistency
PROMPT: There were three apples on the table. Tim took one apple, so now there were
  complex-384 (100.4M)       | playing with a lot of fun. Suddenly, the girl saw that she had to explore the little bit him. She was very happy and wanted
  real-588 (101.9M, pm)      | no food and it was very important to be safe! But then, his mom went home and said goodbye to him. She had made a
  real-768 (162.0M)          | no food. He wanted to be fun, but he didn't know how. Suddenly, a friendly voice came and said "You have

### [grammar-noun] basic grammar, plausible noun
PROMPT: The little dog ran quickly to the
  complex-384 (100.4M)       | park. The sun went for a few days of fun with his mom. After he saw something special and found a big hug, red ball
  real-588 (101.9M, pm)      | rabbit. The lady smiled and said, "We can be kind of you!" And they went home and played together with their toys.
  real-768 (162.0M)          | park. The sun came and saw the big pond, dark and the sky. The birds saw a little bird and flew down. Lily tried to catch

### [causal-umbrella] cause -> object (umbrella)
PROMPT: It started to rain, so Tim went into his room and took his
  complex-384 (100.4M)       | new friend. He gave them a big house with a lot of fun day. Suddenly, Timmy found a little girl that was very
  real-588 (101.9M, pm)      | dog home. He was very happy that he couldn't fly away with him. Suddenly, a little boy came over to play with the
  real-768 (162.0M)          | home. He was sad that he didn't know what to do. Timmy's mom said "No, I will be careful with the

### [hungry-eat] cause -> food
PROMPT: Lily was very hungry, so she went to the kitchen and ate a
  complex-384 (100.4M)       | new friend. Lily felt happy that she wanted to play with her mommy. She said, "I'm sorry I found your best day for you
  real-588 (101.9M, pm)      | beautiful flower. She gave her a big hug and said "Can we play with me?" "I'm sorry I will be careful not to
  real-768 (162.0M)          | beautiful food. She gave it a big hug and said "Can you help me?" The doctor smiled and said goodbye to Lily.

### [sequence] daily-routine sequence
PROMPT: First, Lily woke up in the morning. Then, she brushed her
  complex-384 (100.4M)       | doll to her friends and saw a big house with her. But she was not happy. She liked the special place but they would fly and laugh.
  real-588 (101.9M, pm)      | mom's hand to stay at home with her mom and dad. When they were done, Lily felt very sad. She said that Lily was
  real-768 (162.0M)          | hands on a tree and saw a bird playing with lots of leaves. She took out to her room and said "I'm sorry I didn't know

### [animal-behavior] animal behavior + place
PROMPT: The bird opened its wings and flew over the
  complex-384 (100.4M)       | park. The sun went for a few days of fun with his mom. After a while, Timmy found a little girl that was always
  real-588 (101.9M, pm)      | rabbit. The bird was happy to see it again, but it was too late. "What happened?" asked the little bird, pointing
  real-768 (162.0M)          | sky. The sun came out and the sky was very dark. When it looked, the little bird saw a big bear! It was so

### [emotion-cause] emotion + reason
PROMPT: Lily was very happy because she
  complex-384 (100.4M)       | could not go back. She gave the man to her mom and mommy for being good her. Her mum looked up a picture of a big hug
  real-588 (101.9M, pm)      | had found her friends. She wished they would always play with her toys and share her toys together. They played hide and had a big day together.
  real-768 (162.0M)          | knew that her mum would always be fun to do her with her mom. Once upon a time, there was an ugly bear named Bobo.

### [story-open] story structure, character
PROMPT: Once upon a time, in a small village, there lived a
  complex-384 (100.4M)       | little boy named Lily. Tim loved to play with his friends who had a toy car. One day, Tim found a big, red ball
  real-588 (101.9M, pm)      | little boy named Timmy. One day, he found an old girl named Lily. She went to play with her toys and run to the park.
  real-768 (162.0M)          | little boy named Timmy. One day, he found an apple on the ground and looked sad. He felt very sad. Timmy

### [plural-grammar] plurals + object
PROMPT: The children went to the park and played with all their
  complex-384 (100.4M)       | toys. The sun was very happy that she couldn't be kind of fun. Suddenly, a little girl named Max found a big hug
  real-588 (101.9M, pm)      | toys. The sun was very happy that they couldn't fly away. They were so excited and said, "I will be careful not to
  real-768 (162.0M)          | toys. The sun was very happy, so they started to play a game. Suddenly, a little girl came to the kitchen. She

### [scared-cause] emotion + cause (monster)
PROMPT: Lily was a little scared because the dark room had a
  complex-384 (100.4M)       | new friend. Lily wanted to be fun, but her mom said she could not have the ball for something special and how much it was for her.
  real-588 (101.9M, pm)      | wonderful friend. Lily wanted to be friends, but she said, "You should have to be careful and not play with your toys." Lily was very
  real-768 (162.0M)          | small bag. Lily wanted to be friends, but she didn't want to share her toys with her friend. But then, Lily noticed that

## Long completions (max_new_tokens=80) — story-level fluency / coherence

### complex-384 (100.4M)
PROMPT: Once upon a time, there was a little girl who loved to play outside.
GEN   : One day, Lily went for a walk in the store with her mommy. She looked up and saw that she had a pretty flowers that lived in a beautiful garden! Suddenly, the bird flew away from the park. Lily said, "That's not a little fun!" said Lily. "I want to eat the bird?" Lily said, "It's so sweet
PROMPT: Tim had a new dog. The dog was small and brown.
GEN   : He liked to play with his new friends, but he did not want to find his friend. They were very happy again. They learned that they were not nice for helping. They played on their way home in the park. Once upon a time, there was a big house named Lily. Tom wanted to help her mommy about how to show his mom. But then she would take
PROMPT: It was a sunny day. Lily decided to go to the park with her friends.
GEN   : She found her friends and saw a big house with her. But she went for being good, but they played together every day. Once upon a time, there was a little girl named Lily. She loved to play outside in the park. One day, she went to the woods outside. The bird could not come out of his toy. The bird said that it was a loud noises and had
PROMPT: One day, the little bird found a shiny red apple.
GEN   : The boy said it was very important to be happy and have fun things. After he put on the store with a big hug and tried to be careful not help when they could. They gave them the man in the park. Lily said she was too tired from that they had found him again. The bear's mom came out of their eyes and got sad. She thought she

### real-588 (101.9M, pm)
PROMPT: Once upon a time, there was a little girl who loved to play outside.
GEN   : One day, she went for a walk in the forest with her mommy. She looked up and saw that it was right away! Her mommy said, "No, you can be careful and not find your room. It's very special. Do you want to buy some ice cream?" The dog nodded and they all came over again. Lily was happy and asked her dad if
PROMPT: Tim had a new dog. The dog was small and brown.
GEN   : He liked to play with his toys, but he could not find the ball. He did not know what to do. One day, Tom's mom made him for a walk. She gave him the man back. His mom said, "It's okay, Sam. I found it!" Tim told her that he couldn't get it. He said, "I'm sorry you so happy
PROMPT: It was a sunny day. Lily decided to go to the park with her friends.
GEN   : She asked her mom, "Lily, look, what's wrong? You have to be careful and play with your toy. It is like to be careful with it." Lily smiled and said, "Yes, I will take you home from the world. Do you want to play with us?" Lila felt bad. She said, "I'm sorry, dad.
PROMPT: One day, the little bird found a shiny red apple.
GEN   : The little girl was sad and started to cry. She looked around for her toy, but it had lost her favorite. Her mom said that she wanted to make something new for. Her mom told her that her didn't know what to do. She asked her mom if they could find one. They saw some things inside, but there was an old man who loved his family. He gave

### real-768 (162.0M)
PROMPT: Once upon a time, there was a little girl who loved to play outside.
GEN   : One day, she went for a walk in the forest with her mommy. She saw a beautiful flower that looked like a pretty flower and it wanted to make it feel better. Her mommy asked him why she was so excited. Her mommy said, "Don't worry, we need to be careful when you see something bad." The little bird was very happy and
PROMPT: Tim had a new dog. The dog was small and brown.
GEN   : The dog saw Lily and Ben. "Hello, dog," Mia said. "Can we play with the dog?" They tried to eat their blocks from the big dog. They ran to the dog and felt a little bit him. It wanted the dog too, but he wanted to be friends. He said, "Let's go of the dog!" But his owner did not know
PROMPT: It was a sunny day. Lily decided to go to the park with her friends.
GEN   : She liked to play hide and seek with her friends, but she always shared her toys and her friends. They had fun playing together in the forest. One day, Lily wanted to take his friend the other kids. They found some cookies and put them back in their mouth. Lily said, "This is your teddy bear." The two of their friends were happy and played together for a
PROMPT: One day, the little bird found a shiny red apple.
GEN   : The little girl was sad because she didn't know what to do. She took out the tree and ran away with it. The little boy saw the sun and went down. He asked her if he could find it. Lily said no, but the little boy did not get very angry. The little bird felt bad for the little little bird. The little bird thought of

## Read

All three are undertrained (1 epoch, tinystories PPL ~20) and collapse into
incoherent, grammar-broken text around ~80 tokens. The quality differences
between arms are subtle and within sampling noise. No arm reliably does
arithmetic ("three apples ... took one" -> none answer "two") or retains the
prompt's named object (the "red ball" is dropped by all three). The bigger
real-768 reads marginally more coherent on a couple of prompts, consistent
with its lower loss. Directionally consistent with the loss verdict
(complex not ahead of the param-matched real), but at 1-epoch the gaps are
fine-grained, not dramatic.
