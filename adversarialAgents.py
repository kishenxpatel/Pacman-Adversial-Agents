# adversarialAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).
#
# Modified for use at University of Bath.


from util import manhattanDistance
import pacman
from game import Directions
import random, util

from game import Agent


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.
    """

    def getAction(self, gameState):
        """
        getAction chooses among the best options according to the evaluation
        function.

        getAction takes a GameState and returns some Directions.X
        for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action)
                  for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores))
                       if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are
        better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        return successorGameState.getScore()


def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()


class AdversarialSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    adversarial searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent and AlphaBetaPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(AdversarialSearchAgent):
    """
    Your minimax agent (question 1)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing
        minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """

        "*** YOUR CODE HERE ***"
         # finds Pacman's best move
        def findMax(state, depth, agent):
            bestMove = None
            if state.isWin() or state.isLose() or depth == self.depth*2:
                return self.evaluationFunction(state), None
            else:
                best = -100000
                legalMoves = state.getLegalActions(agent)
                for i in legalMoves:
                    newState = state.generateSuccessor(agent, i)
                    newScore, minState = findMin(newState, depth + 1, agent + 1)
                    if newScore > best:
                        best = newScore
                        bestMove = i
                return best, bestMove

        # finds the ghosts' best move
        def findMin(state, depth, agent):
            bestMove = None
            if state.isWin() or state.isLose() or depth == self.depth*2:
                return self.evaluationFunction(state), None
            else:
                best = 100000
                legalMoves = state.getLegalActions(agent)
                for i in legalMoves:
                    newState = state.generateSuccessor(agent, i)
                    if agent == state.getNumAgents() - 1:
                        newScore, minState = findMax(newState, depth + 1, 0)
                    else:
                        newScore, minState = findMin(newState, depth, agent + 1)
                    if newScore < best:
                        best = newScore
                        bestMove = i
                return best, bestMove

        #returns best move
        value, move = findMax(gameState, 0, self.index)
        return move

        util.raiseNotDefined()


class AlphaBetaAgent(AdversarialSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax with alpha-beta pruning action using self.depth and
        self.evaluationFunction
        """

        "*** YOUR CODE HERE ***"
        alpha = -1000000
        beta = 1000000

        # finds Pacman's best move
        def findMax(state, depth, agent, alpha, beta):
            bestMove = None
            if state.isWin() or state.isLose() or depth == self.depth * 2:
                return self.evaluationFunction(state), None
            else:
                best = -100000
                legalMoves = state.getLegalActions(agent)
                for i in legalMoves:
                    newState = state.generateSuccessor(agent, i)
                    newScore, minState = findMin(newState, depth + 1, agent + 1, alpha, beta)
                    if newScore > best:
                        best = newScore
                        bestMove = i
                    alpha = max(alpha, best)
                    if alpha > beta:
                        break
                return best, bestMove

        # finds the ghosts' best move
        def findMin(state, depth, agent, alpha, beta):
            bestMove = None
            if state.isWin() or state.isLose() or depth == self.depth * 2:
                return self.evaluationFunction(state), None
            else:
                best = 100000
                legalMoves = state.getLegalActions(agent)
                for i in legalMoves:
                    newState = state.generateSuccessor(agent, i)
                    if agent == state.getNumAgents() - 1:
                        newScore, minState = findMax(newState, depth + 1, 0, alpha, beta)
                    else:
                        newScore, minState = findMin(newState, depth, agent + 1, alpha, beta)
                    if newScore < best:
                        best = newScore
                        bestMove = i
                    beta = min(beta, best)
                    if beta < alpha:
                        break
                return best, bestMove

        # returns best move
        value, move = findMax(gameState, 0, self.index, alpha, beta)
        return move

        util.raiseNotDefined()


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 3).

    DESCRIPTION: <write something here so we know what you did>
    
    To start with I used currentGameState as a paramter so that I was able to get the current position of Pacman
    as well as the current score.
    I checked to see if the game state was a win or a loss (and if so, I returned the upper/lower bound depending
    on the state).
    
    For the power pellets (or capsules) I used .getCapsules() to obtain the amount left on the grid, and for the
    amount of food left I used getNumFood(). Other values I used were the number of "normal" ghosts and the number
    of "vulnerable" (scared) ghosts which I appended into separate lists.
    
    I then calculated the minimum Manhattan distance from Pacman's current position to the nearest point of food and the 
    nearest normal and vulnerable ghost. The resulting values were then used in my calculation of the better score.
    
    The current score obtained from the current state was the first value included in the betterScore variable. I
    subtracted all the values above (with positive scalar multipliers) so that they were all negative - in other
    words, the further away Pacman was from a power pellet, food or a vulnerable ghost, the more negative the score
    was. The only exception was the scalar for the normal ghosts; I took the inverse of the value and put it into a 
    fraction with 1 as the numerator so that the closer to a ghost pacman was, the more negative the score, which
    would be detrimental to the final score.
    
    I chose the scalars through trial and error:
    
    closestFood - * 1.2
    foodAmountLeft - * 8.5
    powerPellets - * 19.8
    closestNormalGhost - * 1/() * 2.4
    closestVulnerableGhost - * 3.2
    
    """

    "*** YOUR CODE HERE ***"
    if currentGameState.isWin(): 
        return 1000000
    elif currentGameState.isLose():
        return -1000000
    
    currentScore = scoreEvaluationFunction(currentGameState)
    currentPosition = currentGameState.getPacmanPosition()
    powerPellets = len(currentGameState.getCapsules())
    foodDistances = []
    food = currentGameState.getFood().asList()
    for foodCoordinate in food:
            foodDistances.append(util.manhattanDistance(currentPosition, foodCoordinate))
    closestFood = min(foodDistances)
    foodAmountLeft = currentGameState.getNumFood()
    
    normalGhosts = []
    vulnerableGhosts = []
    for ghost in currentGameState.getGhostStates():
        if ghost.scaredTimer:
             vulnerableGhosts.append(ghost)
        else: 
             normalGhosts.append(ghost)
    
    normalGhostDistances = []
    vulnerableGhostDistances = []      
    if not normalGhosts:
        closestNormalGhost = 100000
    else:
        for ghost in normalGhosts:
            normalGhostDistances.append(util.manhattanDistance(currentPosition, ghost.getPosition()))
        closestNormalGhost = min(normalGhostDistances)

    if not vulnerableGhosts:
        closestVulnerableGhost = 0
    else:
        for ghost in vulnerableGhosts:
            vulnerableGhostDistances.append(util.manhattanDistance(currentPosition, ghost.getPosition()))
        closestVulnerableGhost = min(vulnerableGhostDistances)

    betterScore = currentScore - (1.2 * closestFood) - (2.4 * (1/closestNormalGhost)) - (3.2 * closestVulnerableGhost) - (8.5 * foodAmountLeft) - (19.8 * powerPellets)
                                            
    return betterScore

    util.raiseNotDefined()


