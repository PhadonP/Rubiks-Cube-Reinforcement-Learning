import sys
import os

sys.path.append(os.path.abspath(""))

import pygame
from pygame.locals import *

from config.config import Config
import argparse

import torch.multiprocessing as mp
import time

import torch
from environment.PuzzleN import PuzzleN
from search.BWAS import batchedWeightedAStarSearch
from networks.PuzzleNet import PuzzleNet


class GUI:
    def __init__(self, game):
        self.game = game
        self.rowLength = self.game.rowLength
        self.N = self.game.N
        self.tileSize = 320 // self.rowLength

        self.windowHeight = 480
        self.windowWidth = 640

        self.fps = 30
        self.blank = None

        self.colours = {
            "black": (0, 0, 0),
            "white": (255, 255, 255),
            "blue": (0, 50, 255),
            "turqoise": (3, 54, 73),
            "green": (0, 204, 0),
        }

        self.bgColour = self.colours["turqoise"]
        self.tileColour = self.colours["green"]
        self.textColour = self.colours["white"]
        self.borderColour = self.colours["blue"]
        self.buttonColour = self.colours["white"]
        self.buttonTextColour = self.colours["black"]
        self.messageColour = self.colours["white"]

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
        self.font = pygame.font.Font("freesansbold.ttf", self.fontSize)
        self.titleFont = pygame.font.Font("freesansbold.ttf", self.titleFontSize)

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
            self.windowWidth - 125,
            self.windowHeight - 70,
        )
        self.lookForSolve = False
        self.solving = False
        self.startSolving = False

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
            if self.lookForSolve or self.solving or self.startSolving:
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
                    self.lookForSolve = True
                elif self.scrambleRect.collidepoint(event.pos):
                    self.game.state = self.game.generateScramble(500)

    def drawGame(self, msg):
        self.displaySurf.fill(self.bgColour)
        if msg:
            self.textSurf, self.textRect = self.makeText(
                msg,
                self.font,
                self.messageColour,
                self.bgColour,
                30,
                self.windowHeight - 40,
            )
            self.displaySurf.blit(self.textSurf, self.textRect)

        for tileY in range(len(self.game.state)):
            for tileX in range(len(self.game.state[0])):
                if self.game.state[tileY][tileX] != 0:
                    self.drawTile(tileX, tileY, int(self.game.state[tileY][tileX]))

        left, top = self.getLeftTopOfTile(0, 0)
        width = self.rowLength * self.tileSize
        height = self.rowLength * self.tileSize
        pygame.draw.rect(
            self.displaySurf,
            self.borderColour,
            (left - 5, top - 5, width + 11, height + 11),
            4,
        )

        self.displaySurf.blit(self.titleSurf, self.titleRect)
        self.displaySurf.blit(self.solveSurf, self.solveRect)
        self.displaySurf.blit(self.scrambleSurf, self.scrambleRect)

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
            (left + adjX, top + adjY, self.tileSize, self.tileSize),
        )
        textSurf = self.font.render(str(number), True, self.textColour)
        textRect = textSurf.get_rect()
        textRect.center = (
            left + int(self.tileSize / 2) + adjX,
            top + int(self.tileSize / 2) + adjY,
        )
        self.displaySurf.blit(textSurf, textRect)

    def getLeftTopOfTile(self, tileX, tileY):
        left = self.xMargin + (tileX * self.tileSize) + (tileX - 1)
        top = self.yMargin + (tileY * self.tileSize) + (tileY - 1)
        return (left, top)


if __name__ == "__main__":

    conf = Config("config/puzzle15.ini")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n", "--network", required=True, help="Path of Saved Network", type=str
    )

    args = parser.parse_args()

    loadPath = args.network

    if not os.path.isfile(loadPath):
        raise ValueError("No Network Saved in this Path")

    device = torch.device(0 if torch.cuda.is_available() else "cpu")

    net = PuzzleNet(conf.puzzleSize).to(device)
    net.load_state_dict(torch.load(loadPath)["net_state_dict"])
    net.eval()

    puzzleN = PuzzleN(conf.puzzleSize)
    GUI = GUI(puzzleN)
    solveQueue = mp.Queue()

    while True:  # main game loop
        msg = "Press arrow keys to slide."  # contains the message to show in the upper left corner.

        if GUI.lookForSolve:
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
                    solveQueue,
                ),
            )
            solveProcess.start()
            GUI.lookForSolve = False
            GUI.solving = True

        if not solveQueue.empty():
            (
                moves,
                numNodesGenerated,
                searchItr,
                isSolved,
                solveTime,
            ) = solveQueue.get()

            GUI.solving = False

            if isSolved:
                print("Solved!")
                print("Moves are %s" % "".join(moves))
                print("Solve Length is %i" % len(moves))
                print("Time of Solve is %.3f seconds" % solveTime)
                GUI.startSolving = True
            else:
                print("No Solution Found")
                print("Search time is %.3f seconds" % solveTime)

            print("%i Nodes were generated" % numNodesGenerated)
            print("There were %i search iterations" % searchItr)

            i = 0
            prevTime = time.time()

        if puzzleN.checkIfSolvedSingle(puzzleN.state):
            msg = "Solved!"
        elif GUI.solving:
            msg = "Solving..."
        elif GUI.startSolving:
            msg = "Found a %d move solution in %.2f seconds" % (len(moves), solveTime)

        GUI.drawGame(msg)
        GUI.checkForQuit()
        GUI.checkInput()

        currTime = time.time()

        if GUI.startSolving and currTime - prevTime > 0.25:
            if i == len(moves):
                GUI.startSolving = False
            else:
                prevTime = currTime
                puzzleN.doAction(moves[i])
                i += 1

        pygame.display.update()

        GUI.fpsClock.tick(GUI.fps)
