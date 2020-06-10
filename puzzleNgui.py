import sys
import os
import enum

import pygame
from pygame.locals import *

from config.config import Config
import argparse

import torch.multiprocessing as mp
import time

import torch
from environment.PuzzleN import PuzzleN
from search.BWAS import batchedWeightedAStarSearch
from networks.getNetwork import getNetwork


class GUIState(enum.Enum):
    ready = 0
    startSolving = 1
    solving = 2
    solved = 3
    fastForward = 4
    rewind = 5


class GUI:
    def __init__(self, game):
        self.game = game
        self.rowLength = self.game.rowLength
        self.N = self.game.N
        self.tileSize = 320 // self.rowLength
        self.tileGap = 8

        self.windowHeight = 480
        self.windowWidth = 640

        self.fps = 30
        self.blank = None

        self.colours = {
            "black": (0, 0, 0),
            "white": (255, 255, 255),
            "red": (230, 127, 119),
            "grey": (242, 242, 242)
        }

        self.bgColour = self.colours["grey"]
        self.tileColour = self.colours["red"]
        self.tileTextColour = self.colours["white"]
        self.textColour = self.colours["black"]

        self.xMargin = int(
            (self.windowWidth - (self.tileSize * self.rowLength + (self.rowLength - 1)))
            / 2
        )
        self.yMargin = int(
            (
                self.windowHeight
                - (self.tileSize * self.rowLength + (self.rowLength - 1))
            )
            / 2
        )

        pygame.init()
        self.fpsClock = pygame.time.Clock()
        self.displaySurf = pygame.display.set_mode(
            (self.windowWidth, self.windowHeight)
        )
        pygame.display.set_caption("Slide Puzzle")

        self.fontSize = 20
        self.titleFontSize = 30
        self.font = pygame.font.SysFont("Consolas", self.fontSize)
        self.titleFont = pygame.font.SysFont(
            "Consolas", self.titleFontSize)

        self.titleSurf, self.titleRect = self.makeText(
            str(self.N) + " Puzzle",
            self.titleFont,
            self.textColour,
            None,
            self.xMargin,
            25,
        )
        self.solveSurf, self.solveRect = self.makeText(
            "Solve",
            self.font,
            self.textColour,
            None,
            self.windowWidth - 85,
            self.windowHeight - 40,
        )
        self.scrambleSurf, self.scrambleRect = self.makeText(
            "Scramble",
            self.font,
            self.textColour,
            None,
            self.windowWidth - 119,
            self.windowHeight - 70,
        )

        self.rewindSurf, self.rewindRect = self.makeText(
            "<<",
            self.font,
            self.textColour,
            None,
            self.windowWidth - 190,
            33,
        )

        self.backwardSurf, self.backwardRect = self.makeText(
            "<",
            self.font,
            self.textColour,
            None,
            self.windowWidth - 140,
            33,
        )

        self.forwardSurf, self.forwardRect = self.makeText(
            ">",
            self.font,
            self.textColour,
            None,
            self.windowWidth - 100,
            33,
        )
        self.fastForwardSurf, self.fastForwardRect = self.makeText(
            ">>",
            self.font,
            self.textColour,
            None,
            self.windowWidth - 60,
            33,
        )

        self.state = GUIState.ready

        self.moves = []
        self.moveIdx = 0

    def checkForQuit(self):
        for event in pygame.event.get(QUIT):  # get all the QUIT events
            pygame.quit()
            sys.exit()  # terminate if any QUIT events are present
        for event in pygame.event.get(KEYUP):  # get all the KEYUP events
            if event.key == K_ESCAPE:
                pygame.quit()
                sys.exit()  # terminate if the KEYUP event was for the Esc key
            pygame.event.post(event)  # put the other KEYUP event objects back

    def checkInput(self):
        for event in pygame.event.get():  # event handling loop
            if not (self.state == GUIState.ready or self.state == GUIState.solved):
                return
            if event.type == KEYUP:
                # check if the user pressed a key to slide a tile
                if event.key in (K_LEFT, K_a):
                    self.game.doAction("L")
                elif event.key in (K_RIGHT, K_d):
                    self.game.doAction("R")
                elif event.key in (K_UP, K_w):
                    self.game.doAction("U")
                elif event.key in (K_DOWN, K_s):
                    self.game.doAction("D")
            elif event.type == MOUSEBUTTONUP:
                if self.solveRect.collidepoint(event.pos):
                    self.state = GUIState.startSolving
                elif self.scrambleRect.collidepoint(event.pos):
                    self.game.state = self.game.generateScramble(500)
                    self.state = GUIState.ready
                elif self.rewindRect.collidepoint(event.pos):
                    self.state = GUIState.rewind
                elif self.backwardRect.collidepoint(event.pos):
                    self.backwardMove()
                elif self.forwardRect.collidepoint(event.pos):
                    self.forwardMove()
                elif self.fastForwardRect.collidepoint(event.pos):
                    self.state = GUIState.fastForward

    def drawGame(self, msg):
        self.displaySurf.fill(self.bgColour)
        if msg:
            self.textSurf, self.textRect = self.makeText(
                msg,
                self.font,
                self.textColour,
                self.bgColour,
                30,
                self.windowHeight - 40,
            )
            self.displaySurf.blit(self.textSurf, self.textRect)

        for tileY in range(len(self.game.state)):
            for tileX in range(len(self.game.state[0])):
                if self.game.state[tileY][tileX] != 0:
                    self.drawTile(tileX, tileY, int(
                        self.game.state[tileY][tileX]))

        self.displaySurf.blit(self.titleSurf, self.titleRect)
        self.displaySurf.blit(self.solveSurf, self.solveRect)
        self.displaySurf.blit(self.scrambleSurf, self.scrambleRect)
        if self.state == GUIState.solved:
            if self.moveIdx > 0:
                self.displaySurf.blit(self.rewindSurf, self.rewindRect)
                self.displaySurf.blit(self.backwardSurf, self.backwardRect)
            if self.moveIdx < len(self.moves):
                self.displaySurf.blit(self.forwardSurf, self.forwardRect)
                self.displaySurf.blit(
                    self.fastForwardSurf, self.fastForwardRect)

    def makeText(self, text, font, colour, bgColour, top, left):
        # create the Surface and Rect objects for some text.
        textSurf = font.render(text, True, colour, bgColour)
        textRect = textSurf.get_rect()
        textRect.topleft = (top, left)
        return (textSurf, textRect)

    def drawTile(self, tileX, tileY, number, adjX=0, adjY=0):
        left, top = self.getLeftTopOfTile(tileX, tileY)
        pygame.draw.rect(
            self.displaySurf,
            self.tileColour,
            (left + adjX + self.tileGap // 2, top + adjY + 2,
             self.tileSize - self.tileGap, self.tileSize - self.tileGap)
        )
        textSurf = self.font.render(str(number), True, self.tileTextColour)
        textRect = textSurf.get_rect()
        textRect.center = (
            left + int(self.tileSize / 2) + adjX,
            top + int(self.tileSize / 2) + adjY,
        )
        self.displaySurf.blit(textSurf, textRect)

    def getLeftTopOfTile(self, tileX, tileY):
        left = self.xMargin + (tileX * self.tileSize)
        top = self.yMargin + (tileY * self.tileSize)
        return (left, top)

    def setMoves(self, moves):
        self.moves = moves
        self.moveIdx = 0

    def forwardMove(self):
        if self.moveIdx < len(self.moves):
            self.game.doAction(self.moves[self.moveIdx])
            self.moveIdx += 1
        else:
            self.state = GUIState.solved

    def backwardMove(self):
        if self.moveIdx > 0:
            forwardMove = self.moves[self.moveIdx - 1]
            backwardMoveDict = {"R": "L",
                                "U": "D",
                                "D": "U",
                                "L": "R"
                                }
            backwardMove = backwardMoveDict[forwardMove]

            self.game.doAction(backwardMove)
            self.moveIdx -= 1
        else:
            self.state = GUIState.solved


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n", "--network", required=True, help="Path of Saved Network", type=str
    )

    parser.add_argument(
        "-c", "--config", required=True, help="Path of Config File", type=str
    )

    args = parser.parse_args()

    conf = Config(args.config)

    loadPath = args.network

    if not os.path.isfile(loadPath):
        raise ValueError("No Network Saved in this Path")

    device = torch.device(0 if torch.cuda.is_available() else "cpu")

    net = getNetwork(conf.puzzle, conf.networkType)(conf.puzzleSize)
    net.load_state_dict(torch.load(loadPath))
    net.eval()

    puzzleN = PuzzleN(conf.puzzleSize)
    GUI = GUI(puzzleN)
    solveQueue = mp.Queue()

    while True:  # main game loop
        # contains the message to show in the upper left corner.

        if GUI.state == GUIState.startSolving:
            solveProcess = mp.Process(
                target=batchedWeightedAStarSearch,
                args=(
                    puzzleN.state,
                    conf.depthWeight,
                    conf.numParallel,
                    puzzleN,
                    net,
                    device,
                    conf.maxSearchItr,
                    True,
                    solveQueue,
                ),
            )
            solveProcess.start()
            GUI.state = GUIState.solving

        if not solveQueue.empty():
            (
                moves,
                numNodesGenerated,
                searchItr,
                isSolved,
                solveTime,
            ) = solveQueue.get()

            if isSolved:
                print("Solved!")
                print("Moves are %s" % "".join(moves))
                print("Solve Length is %i" % len(moves))
                print("Time of Solve is %.3f seconds" % solveTime)
                GUI.state = GUIState.solved
                GUI.setMoves(moves)
            else:
                print("No Solution Found")
                print("Search time is %.3f seconds" % solveTime)
                GUI.state = GUIState.ready

            print("%i Nodes were generated" % numNodesGenerated)
            print("There were %i search iterations" % searchItr)

            prevTime = time.time()

        if puzzleN.checkIfSolvedSingle(puzzleN.state):
            msg = "Solved!"
        elif GUI.state == GUIState.solving:
            msg = "Solving..."
        elif GUI.state != GUIState.ready:
            msg = "Found a %d move solution in %.2f seconds" % (
                len(moves), solveTime)
        else:
            msg = "Press arrow keys to slide."

        GUI.drawGame(msg)
        GUI.checkForQuit()
        GUI.checkInput()

        currTime = time.time()

        if (GUI.state == GUIState.fastForward or GUI.state == GUIState.rewind) and currTime - prevTime > 0.25:
            prevTime = currTime
            if GUI.state == GUIState.fastForward:
                GUI.forwardMove()
            else:
                GUI.backwardMove()

        pygame.display.update()

        GUI.fpsClock.tick(GUI.fps)
