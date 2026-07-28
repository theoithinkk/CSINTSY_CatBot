# Major Course Output 3: CatBot

**Release Date:** October 24, 2025  
**Due Date:** November 28, 2025 (Friday), 11:59 PM

---

## 1. Project Description

In this project, students will apply the concepts and theories they have learned in the course to develop a simple cat-catching bot that can adapt to the behaviors of different cats. Through this project, students should be able to demonstrate the following learning outcomes:

- **LO3.** Collaboratively design and implement machine learning models for supervised and unsupervised tasks
- **LO5.** Articulate ideas and present results in correct technical written and oral English.

---

## 2. Project Specifications

This section contains the specifications for this major course output.

### 2.1 Overview

The cats in campus are on the loose! DLSU PUSA has enlisted CCS students to help catch them. Of course, we can't expect CCS students to go running around the campus just to chase some cats. Instead, your group decides to build a bot that can do the job. In this project, you will implement the learning component of a bot called **CatBot**. CatBot is a bot that can catch cats in a 2-D grid. The bot must use reinforcement learning to automatically learn the behavior of different cats, and adapt its strategy accordingly.

### 2.2 The World

The world is an **8 × 8 grid**. In a given scenario, CatBot and a single cat will start at predefined cells in the grid. CatBot has five possible actions:
- Move up
- Move down
- Move left
- Move right
- Stay in place

The goal of CatBot is to catch the cat by ending up in the same cell as the cat. Every time CatBot makes an action, the cat will also have a chance to perform one action. Different cats will behave differently according to their personality.

### 2.3 Catching Cats

You might think that catching the cats is just about following the shortest path to the cat. However, this is not always the case. Some cats are very playful, and will always run away if you just go straight for them. Therefore, a shortest-path strategy will not always work. 

Effective cat-catching requires learning the personality and behavior of the cat, knowing how to appease them until you can catch them. Therefore, CatBot must be implemented with a reinforcement learning component that can:
- Explore interactions with a given cat
- Learn how it responds to different actions
- Adapt its strategy accordingly

### 2.4 The Cats

There will be **ten different cats** to catch. Each cat has a different personality and behavior. Your bot must use the same learning algorithm to adapt its behavior to any type of cat. 

In this project, you will be provided the specifications of the first five cats so you can experiment with your learning algorithm. The remaining five cats will be hidden and will be used for testing your bot during evaluation. CatBot must be capable of automatically learning the behavior of these hidden cats, even if you don't have access to their specifications during development.

#### Cat List

| Cat Name | Photo | Personality | Behavior Description |
|----------|-------|-------------|----------------------|
| Batmeow (batmeow) | [Image] | Sleepy | Batmeow is too sleepy to move and just stays in place the entire time. |
| Mittens (mittens) | [Image] | Energetic | Mittens had too much sugar and just moves around randomly. |
| Paotsin (paotsin) | [Image] | Playful | The more you chase Paotsin, the more he runs away. |
| Squiddyboi (squiddyboi) | [Image] | Athletic | Squiddyboi can jump over you, and will not hesitate to do so when threatened. |
| Peekaboo (peekaboo) | [Image] | Ninja | Peekaboo can teleport to any edge of the map, but some say he has a weak side... |
| Spidercat (spidercat) | [Image] | Stalker | *The behavior will remain a mystery until your project has been submitted.* |
| Cheddar (cheddar) | [Image] | Intelligent | *The behavior will remain a mystery until your project has been submitted.* |
| Pumpkin Pie (pumpkinpie) | [Image] | Angry | *The behavior will remain a mystery until your project has been submitted.* |
| Milky (milky) | [Image] | Magical | *The behavior will remain a mystery until your project has been submitted.* |
| Taro (taro) | [Image] | Enzo | *The behavior will remain a mystery until your project has been submitted.* |
| Trainer Cat (trainer) | [Image] | N/A | This is a special robot cat. You can program its behavior to help you test the effectiveness of your learning algorithm in custom scenarios. |

---

## 3. Starter Code

You are provided starter codes for this project. You are expected to use **Python 3.12** for this project. Earlier versions of Python may not be compatible with the OpenAI Gymnasium package used in this project.

### File Descriptions

- **cat_env.py** - Contains the implementation of the custom CatBot environment using the OpenAI Gymnasium framework. You are NOT allowed to modify this file in any way, except for the implementation of the Trainer cat.
- **utility.py** - Contains helper functions for running the custom CatBot environment. You are also NOT allowed to modify this file in any way.
- **play.py** and **bot.py** - Scripts for running CatBot in freeplay mode and bot mode, respectively. You are also NOT allowed to modify these files.
- **training.py** - Contains the skeleton code for the reinforcement learning algorithm of CatBot. You are expected to implement the learning algorithm in this file.

### Getting Started

It is recommended that you familiarize yourself first with the environment by running the `play.py` script. This runs the game in freeplay mode, where you can manually control CatBot using the keyboard. By default, the script loads the Batmeow cat scenario.

**To run a different cat scenario:**
```bash
python play.py --cat mittens
```

Try to play around with the different cat types manually to understand their behaviors.

**To train and run the bot:**
```bash
python bot.py --cat paotsin
```

**To visualize the bot's behavior during training:**
```bash
python bot.py --cat paotsin --render 100
```

By default, the `train_bot` function does not have any training logic, which means that the bot will not learn anything. Your primary task is to implement a Q-learning algorithm by filling in the missing sections in `training.py`.

---

## 4. Implementing the Reinforcement Learning Algorithm

You are expected to implement a reinforcement learning algorithm in the `training.py` file. For this project, Q-learning is enough, but you are free to explore other algorithms if you wish.

### 4.1 States and Actions Representation

The CatBot environment is an OpenAI Gymnasium environment. It is highly recommended that you follow online tutorials on OpenAI Gymnasium to familiarize yourself with how to interact with an environment and how to implement reinforcement learning algorithms.

**Recommended tutorials:**
- OpenAI Gymnasium - Training an Agent Tutorial
- Q-Learning for Beginners

#### State Representation

- There are a total of **10,000 possible states** (some of which are unused)
- Each state is represented as an **integer from 0 to 9999**
- The first two digits represent the row and column of CatBot
- The last two digits represent the row and column of the cat
- Example: state `2305` means CatBot is at row 2, column 3, while the cat is at row 0, column 5
- Rows and columns are 0-indexed
- Note: Because the grid is only 8 × 8, some states are never used (e.g., state 8800)

#### Action Representation

- Actions are represented as integers from 0 to 4:
  - `0` = Move up
  - `1` = Move down
  - `2` = Move left
  - `3` = Move right
  - `4` = Stay in place

### 4.2 Q-Learning

You are expected to implement the Q-learning algorithm in the `train_bot` function in `training.py`. In general, each episode of training should cover the following:

1. Reset the environment to start a new episode
2. Decide whether to explore or exploit
3. Take the action and observe the next state
4. Since this environment doesn't give rewards, compute reward manually
5. Update the Q-table accordingly based on agent's rewards

#### Important Restrictions

- You are NOT allowed to modify any part of the code outside the designated areas
- The training is fixed to **5000 episodes** (you cannot change this)
- The entire training process must be complete in at most **20 seconds** (this does not include animation time)
- You are free to design your own:
  - Reward structure
  - Exploration strategy
  - Hyperparameters
  - Other aspects of the learning algorithm

#### Important Notes

- By default, the CatBot environment does not give out any rewards
- You must design your own reward structure to effectively train CatBot
- You are NOT supposed to modify `cat_env.py` to implement the reward structure
- Instead, implement the reward mechanism manually in `training.py`
- You can implement your own custom cat behavior by modifying the Trainer cat in `cat_env.py` to test your learning algorithm
- You are NOT allowed to modify the behavior of the other cats or any other parts of `cat_env.py`

---

## 5. Evaluation

To get perfect credit for the functionality, your bot must be able to catch all ten cats. Furthermore, in each scenario, your bot will only have a maximum of **60 moves** to catch the cat. This restriction ensures that your bot is truly capable of learning the cat behavior and strategies, and not just blindly trying out actions until it gets lucky. If the bot exceeds 60 moves in a given scenario, the cat will be considered uncaught.

---

## 6. Report

In addition to the CatBot code, you are required to write a report documenting the implementation of the project. The report should contain the following:

### 6.1 Reinforcement Learning Algorithm
A section describing the implementation of the Q-Learning algorithm, including:
- The reward structure
- Exploration strategy
- Hyperparameters
- Any other relevant details

You should also justify your design choices and explain how they contribute to the effectiveness of the learning algorithm.

### 6.2 Evaluation and Performance
A section containing an empirical evaluation of the performance of your bot on different cat scenarios. You should provide:
- Quantitative results (e.g., average number of steps taken to catch each cat after training)
- Qualitative observations on the behavior of the bot
- Graphs or tables to illustrate your results

### 6.3 Challenges
A discussion on the difficulties encountered in developing the bot. What are the challenges that make it difficult for algorithms to adapt to new behavior?

### 6.4 Table of Contributions
A table showing the contributions of each member to the project.

### Report Guidelines

- **Maximum length:** 4 pages (A4 sized paper)
- If there is a strong need for more pages, ask permission from the instructor and justify it
- **Formatting:** No restriction, as long as it is readable
- **Content:** Be concise and avoid repeating already-known definitions
- The bulk should contain your group's personal ideas, insights, and implementations
- **AI Use:** You are allowed to use generative AI
- The report should be an accurate reflection of your group's personal understanding and engagement with the project

---

## 7. Deliverables

There are two deliverables for this project:

1. **ZIP file** - Containing the source files for the project
   - Must include all source files needed to run the bot
   - Must include `training.py` with the `train_bot` function implemented

2. **PDF file** - Containing the report

---

## 8. Academic Honesty

The honesty policy applies. Please note that you are NOT allowed to:
- Borrow and/or copy-and-paste in full or in part any existing related program code from the internet or other sources
- Copy from printed materials like books or source codes by other people that are not online

**Violating this policy is a serious offense** in De La Salle University and will result in a grade of 0.0 for the whole course.

### Important Notes

- The point of this project is for all members to learn something and increase their appreciation of the concepts covered in class
- Each member is expected to be able to explain the different aspects of their submitted work, whether part of their contributions or not
- Failure to do this will be interpreted as a failure of the learning goals and will result in a grade of 0 for that member

---

## 9. Credits

Some images used in this project were obtained from DLSU PUSA's cat directory. The detailed list of credits can be found in the `credits.txt` file included in the package. Images of the cats are credited to their respective photographers as indicated in the directory, and their use in this project falls under fair use.

**Please do not redistribute any part of this project publicly or commercially.**

---

## 10. Generative AI Use Declaration

Some aspects of this project were prepared with the help of Gemini 2.5 and Claude Sonnet 3.5, namely:
- Assistance in preparing the starter code
- Generating images for the CatBot

The idea and design of the project is the original idea of the instructor. Generative AI was simply used to make its production more efficient. Any code or images generated by generative AI has been thoroughly reviewed, modified, and synthesized to align with the instructor's intentions and objectives.

The instructor takes full accountability for all the contents of this package.

**Please note:** Students are allowed to use generative AI for code generation or text generation in this project as long as it is properly documented and used to learn from.
