# 76M Scaling Diagnostics

checkpoint: `D:\CodexLLM\research1_longseq\runs\wave10_3080_lowrank_conv_memory_76m_to5b_fresh_after2b_smartlr_20260615\checkpoint.pt`

## Continuation Preference Tests

### machine_learning_definition

Prompt: `Question: What is machine learning? Answer:`

- `generic_web` mean_nll=2.3789: ' The following information is provided by the Department of Education.'
- `observed_style` mean_nll=2.5253: ' The best way to learn a little about the next step is to have them practice.'
- `expected` mean_nll=4.4292: ' Machine learning is a method where computers learn patterns from data.'

### python_add_function

Prompt: `Python function to add two numbers:
`

- `generic_web` mean_nll=2.9365: 'This section describes the procedure in the following table.'
- `observed_style` mean_nll=3.5445: 'A variable in the first determines the number of functions of order variables.'
- `expected` mean_nll=4.7707: 'def add(a, b):\n    return a + b\n'

### validation_loss_definition

Prompt: `In plain English, validation loss measures`

- `observed_style` mean_nll=2.9384: ' are the result of reduced yields of new products.'
- `generic_web` mean_nll=3.1024: ' the data that are available in the report.'
- `expected` mean_nll=5.7643: ' how well the model predicts unseen validation data.'

### sky_blue_answer

Prompt: `A child asked why the sky is blue. The teacher said`

- `observed_style` mean_nll=2.9749: " just as a child, you said. Now, you can't see what's going on."
- `generic_web` mean_nll=3.9942: ' the following program was designed for preschool students.'
- `expected` mean_nll=4.6154: ' sunlight is scattered by air molecules, and blue light scatters more.'

## Decode Sweep

### greedy | 'Question: What is machine learning? Answer:'

Question: What is machine learning? Answer:
- What is the difference between the two?
- What is the difference between the two?
- What is the difference between the two?
- What is the difference between the two?
- What is the difference between the two?
- What is the difference between the two?
- What is the difference between the two?
- What is the difference between the two?

### greedy | 'Python function to add two numbers:\n'

Python function to add two numbers:
- The first one is to use the
- and the second
- the second
- the second
- the second
- the second
- the second
- the second
- the second
- the second
- the second
- the second
- the second
- the second
- the second
- the second
- the second
- the second
- the

### greedy | 'In plain English, validation loss measures'

In plain English, validation loss measures are not a factor.
The problem is that the data are not reliable. The data are not reliable.
The data are not reliable.
The data are not reliable.
The data are not accurate.
The data are not accurate.
The data are not accurate.
The data are not accurate.
The data are not accurate.
The data are not accurate.
The

### greedy | 'A child asked why the sky is blue. The teacher said'

A child asked why the sky is blue. The teacher said, "I am a teacher, I am a teacher, and I am a teacher."
"I am a teacher. I am a teacher. I am a teacher. I am a teacher. I am a teacher. I am a teacher. I am a teacher. I am a teacher. I am a teacher. I am a teacher. I am a teacher. I am a teacher.

### greedy | 'To debug a failing training run, first'

To debug a failing training run, first and foremost, to be able to do so.
The first step in the process is to use the first step in the process.
The first step is to use the first step in the process.
The first step is to use the first step in the process.
Step 3: The process begins with the first step.
Step 3: Step 3: Step 3: Step 3:

### greedy | 'Write a polite email asking for more information:'

Write a polite email asking for more information:
- What is the purpose of the document?
- What is the purpose of the document?
- What is the purpose of the document?
- What is the purpose of the document?
- What is the purpose of the document?
- What is the purpose of the document?
- What is the purpose of the document?
- What is the purpose of the document?

### low_temp_top20 | 'Question: What is machine learning? Answer:'

Question: What is machine learning? Answer:
1. What is the difference between a non-negative and a negative number?
3. What is the difference between a positive and negative?
3. What is the difference between a positive and negative one?
4. What is the difference between the two?
3. What is the difference between a positive and negative positive?
5. What is the difference between a positive and

### low_temp_top20 | 'Python function to add two numbers:\n'

Python function to add two numbers:
<std::string(1, 1, 2, 3, 3, 4, 4, 4, 5, 5, 6, 7, 10, 11, 12, 9, 9, 12, 12, 16, 16, 15, 19, 30, 32, 31, 32, 34, 36, 48, 48, 48, 36, 36, 36, 31, 34, 42

### low_temp_top20 | 'In plain English, validation loss measures'

In plain English, validation loss measures are the most important factor.
The most common use of the word "sugar" is that it is a "dummy" sound, which means "sugar." The word "d" is a very old word for "sugar," meaning "sugar."
The word "drink" is used in the same way as "to rub" or "to."
In the

### low_temp_top20 | 'A child asked why the sky is blue. The teacher said'

A child asked why the sky is blue. The teacher said, "I have a good eye for the eye."
"I'm going to be able to see what I'm saying, but I think it's a good idea to have a good night's sleep," he said. "I think that's really cool. I'm not going to get a better night's sleep, but I'm not sure if I'm going to have to go to sleep

### low_temp_top20 | 'To debug a failing training run, first'

To debug a failing training run, first in the morning and after the first day of the game, the first time, the second time, and the second time, the second stage of the game, the second stage, the second stage, and the second stage.
In the second phase, the player is given a first and second stage. The second stage is the second stage, the second stage, and the second stage begins. The

### low_temp_top20 | 'Write a polite email asking for more information:'

Write a polite email asking for more information:
- The author is the author.
- The author is a member of the author.
- The author is a member of the author.
- The author is a good source of information.
- The author is a writer.
- The author is a writer.
- The author is the author.
- The author is the author.
- The reader is not a

### prior_sample_top64 | 'Question: What is machine learning? Answer:'

Question: What is machine learning? Answer: yes, they’re asking you to go back and turn it all out?’ They could not see anything but the answer.
NARRATOR: In a way, this is a huge problem, you’ll need to get a new job. The cost is really big and the results are clear.
And it is the reason that this system is going to be a lot

### prior_sample_top64 | 'Python function to add two numbers:\n'

Python function to add two numbers:
<std::string*', 's' and __new'' will be
#include <stdonto.h> >>> | |
To make the loop-level an iterator-separated, i can call the function
of any of its above function.
As a result,
this is a special form that is
In the case of the
, you may use this expression to

### prior_sample_top64 | 'In plain English, validation loss measures'

In plain English, validation loss measures are the greatest risk, but our findings are far out of proportion to the fact that the data are not reliable.
You may be thinking about how much CO2 reaching the bottom of the chart is. The graph shows the same correlation between the two graphs. The graphs show the number of cases that were measured by two factors: temperature, temperature, precipitation, air temperature, and heat. The difference

### prior_sample_top64 | 'A child asked why the sky is blue. The teacher said'

A child asked why the sky is blue. The teacher said: "Here are some facts you need to know about the picture but you are going to learn a lot."
6) There is no reason that the age of the student is not that simple. "It has a special purpose, and so I often do not know what to tell." Instead, it is good to have a good understanding of the topic and its contents. That is, it is

### prior_sample_top64 | 'To debug a failing training run, first'

To debug a failing training run, first in the hospital, or in the hospital for 3rd degree.
Called the patient and one in the hospital, or in the case of the last, is a series of emergency rooms available, and as the new surgical facility, the patient's physician orders the patient to perform this operation.
The surgeon's specialties are often the most common surgical treatments of any type of spinal surgery surgery.

### prior_sample_top64 | 'Write a polite email asking for more information:'

Write a polite email asking for more information:
- The news:
- the message is being tweeted and the phone goes down.
- A link between what you are hearing, and so on.
- When you're talking, you can tell if you want to see what you are doing.
- You can do it.
- You have to find it.
- The reason you are posting isn't a good idea.
