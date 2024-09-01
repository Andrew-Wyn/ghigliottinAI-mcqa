# ghigliottinAI-mcqa

Starting from public released games of "La Ghigliottina", founded on two old edition of EVALITA:

- EVALITA-2020: https://ghigliottin-ai.github.io/
- EVALITA-2018: https://nlp4fun.github.io/

We collected 615 different games, from TV game show and from Boardgame.

"La Ghigliottina" is a complex game, to be solved, it needs a very large comprehension of the italian cultural knowledge.

Moreover, the game as it is, is not well posed, we don't have any guarantee that the proposed solution by the authors of the game is unique, this makes impossible to properly evaluate the models to solve that game.
To overcome this intrinsic limitation we and motivated by the strong complexity of the game itself to be solved even if the single sample has a single solution, we reframed the model as a multi-choice question answering.

For each sample game, we created three different distractors, in that case the game can be posed in the following way:

- Hints: w1, w2, w3, w4, w5
- choices: d1, d2, d3, solution.

Where d1, d2, d3 (the distractors) are choosen to be not compatible with the solution definition of the game.

Each distractor was choosen to be related with the solution and 3 random hints (high cosine similarity with fasttext embeddings) and uncorrelated with the other 2 hints (low cosine similarity). With this setting we aim to create distractors that are not completelly random, but moreover that are not possible game solutions.