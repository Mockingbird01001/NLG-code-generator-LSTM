import tkinter
class pvp_app():
    def __init__(self):
        tkinter.Frame.__init__(self)
    '''
        for i in WinCombo:
            if((self.bState[i[0]]+self.bState[i[1]]+self.bState[i[2]])==3):
                self.start_label.config(text="Player 1 won!!")
                self.GameDone=TRUE
                for k in range(1,10):
                    self.button_lst[k].configure(state=DISABLED)
                break
            elif((self.bState[i[0]]+self.bState[i[1]]+self.bState[i[2]])==-3):
                self.start_label.config(text="Player 2 won!!")
                self.GameDone=TRUE
                for k in range(1,10):
                    self.button_lst[k].configure(state=DISABLED)
                break
            else:
                if not (0 in self.bState):
                    self.start_label.config(text="Match drawn!!")
                    self.GameDone=TRUE
                    for k in range(1,10):
                        self.button_lst[k].configure(state=DISABLED)
                    break
    '''
    def PlayerGame(self):
        self.pvp_button.grid_forget()
        self.back_s_button.grid_forget()
        self.startgame_label.grid_forget()
        self.start_label = tkinter.Label(self, text="Who starts the game??")
        self.start_label.grid(row=0, columnspan=11)
        self.p1_button = tkinter.Button(self, text="Player 1", fg="blue",
                                        command=lambda: self.startGame(
                                            tkinter.TRUE, tkinter.FALSE))
        self.p1_button.grid(row=0, column=0, columnspan=2)
        self.p2_button = tkinter.Button(self, text="Player 2", fg="blue",
                                        command=lambda: self.startGame(
                                            tkinter.FALSE, tkinter.TRUE))
        self.p2_button.grid(row=0, column=8, columnspan=2)
        self.back_p_button = tkinter.Button(self, text="Back", fg="blue",
                                            bg="green",
                                            command=self.back_r_button,
                                            height=1,
                                            width=2)
        self.back_p_button.grid(row=12, column=5)
        self.button_lst = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                           [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                           [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                           [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                           [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                           [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                           [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                           [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                           [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                           [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]
        for i in range(1, 10):
            for j in range(1, 10):
                self.button_lst[i][j] = tkinter.Button(self, text=" ",
                                                       fg="yellow",
                                                       bg="red",
                                                       state=tkinter.DISABLED,
                                                       command=lambda x=i,
                                                                      y=j:
                                                       self.handle_button(
                                                           x, y), height=1,
                                                       width=2,
                                                       font=("Purisa", 18))
        L = [[[1, 2, 0], [2, 2, 1], [3, 2, 2],
              [4, 3, 0], [5, 3, 1], [6, 3, 2],
              [7, 4, 0], [8, 4, 1], [9, 4, 2]],
             [[1, 2, 4], [2, 2, 5], [3, 2, 6],
              [4, 3, 4], [5, 3, 5], [6, 3, 6],
              [7, 4, 4], [8, 4, 5], [9, 4, 6]],
             [[1, 2, 7], [2, 2, 8], [3, 2, 9],
              [4, 3, 7], [5, 3, 8], [6, 3, 9],
              [7, 4, 7], [8, 4, 8], [9, 4, 9]],
             [[1, 5, 0], [2, 5, 1], [3, 5, 2],
              [4, 6, 0], [5, 6, 1], [6, 6, 2],
              [7, 7, 0], [8, 7, 1], [9, 7, 2]],
             [[1, 5, 4], [2, 5, 5], [3, 5, 6],
              [4, 6, 4], [5, 6, 5], [6, 6, 6],
              [7, 7, 4], [8, 7, 5], [9, 7, 6]],
             [[1, 5, 7], [2, 5, 8], [3, 5, 9],
              [4, 6, 7], [5, 6, 8], [6, 6, 9],
              [7, 7, 7], [8, 7, 8], [9, 7, 9]],
             [[1, 8, 0], [2, 8, 1], [3, 8, 2],
              [4, 9, 0], [5, 9, 1], [6, 9, 2],
              [7, 10, 0], [8, 10, 1], [9, 10, 2]],
             [[1, 8, 4], [2, 8, 5], [3, 8, 6],
              [4, 9, 4], [5, 9, 5], [6, 9, 6],
              [7, 10, 4], [8, 10, 5], [9, 10, 6]],
             [[1, 8, 7], [2, 8, 8], [3, 8, 9],
              [4, 9, 7], [5, 9, 8], [6, 9, 9],
              [7, 10, 7], [8, 10, 8], [9, 10, 9]]]
        count = 1
        for j in L:
            for i in j:
                self.button_lst[count][i[0]].grid(row=i[1], column=i[2])
            count += 1
    def startGame(self, p1, p2):
        self.p1_button.config(state=tkinter.DISABLED)
        self.p2_button.config(state=tkinter.DISABLED)
        self.back_p_button.config(state=tkinter.DISABLED)
        for i in range(1, 10):
            for j in range(1, 10):
                self.button_lst[i][j].config(state=tkinter.NORMAL)
        if (p1 == tkinter.TRUE):
            self.start_label.config(text="Player 1 starts with X")
            self.turn = tkinter.TRUE
        else:
            self.start_label.config(text="Player 2 starts with O")
            self.turn = tkinter.FALSE
        self.bState = [[10, 10, 10, 10, 10, 10, 10, 10, 10, 10],
                       [10, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [10, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [10, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [10, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [10, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [10, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [10, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [10, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [10, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
        self.bSum = [[0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0], ]
        self.fState = [10, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.bWin = [10]
        self.GameDone = tkinter.FALSE
    def handle_button(self, i, j):
        if (self.turn == tkinter.TRUE):
            self.button_lst[i][j].config(text="X", state=tkinter.DISABLED)
            self.turn = tkinter.FALSE
            self.bState[i][j] = 1
            if i in self.bWin:
                for a in range(1, 10):
                    for b in range(1, 10):
                        self.button_lst[a][b].config(state=tkinter.DISABLED,
                                                     bg="red")
                for a in range(1, 10):
                    self.button_lst[j][a].config(state=tkinter.NORMAL,
                                                 bg="blue")
            else:
                for a in self.bWin:
                    for b in range(1, 10):
                        self.button_lst[a][b].config(state=tkinter.DISABLED,
                                                     bg="red")
                for a in range(1, 10):
                    self.button_lst[j][a].config(state=tkinter.NORMAL,
                                                 bg="blue")
        else:
            self.button_lst[i][j].config(text="O", state=tkinter.DISABLED)
            self.turn = tkinter.TRUE
            self.bState[i][j] = -1
            for a in range(1, 10):
                for b in range(1, 10):
                    self.button_lst[a][b].config(state=tkinter.DISABLED,
                                                 bg="red")
            for a in range(1, 10):
                self.button_lst[j][a].config(state=tkinter.NORMAL, bg="blue")
        self.check()
    def check(self):
        self.sum()
        count = 0
        for i in self.bSum:
            count += 1
            for j in i:
                if (j == 3):
                    self.fState[count] = 1
                    self.bWin.append(count)
                elif (j == -3):
                    self.fState[count] = -1
                    self.bWin.append(count)
                else:
                    check = 0
                    for i in self.bState:
                        for j in i:
                            if j != 0 and j != 10:
                                check = 1
                    if check == 0:
                        for a in range(1, 10):
                            for b in range(1, 10):
                                self.button_lst[a][b].config(
                                    state=tkinter.DISABLED)
                        break
        if self.GameDone == tkinter.TRUE:
            self.message = tkinter.messagebox.askquestion(" ",
                                                          "Do you want to "
                                                          "play the "
                                                          "game again?")
            if self.message == "no":
                self.QuitGame()
            else:
                self.p1_button.config(state=tkinter.NORMAL)
                self.p2_button.config(state=tkinter.NORMAL)
                self.back_p_button.config(state=tkinter.NORMAL)
                self.start_label.config(text="Who will start the game?")
                for a in range(1, 10):
                    for b in range(1, 10):
                        self.button_lst[a][b].config(state=tkinter.NORMAL)
    def sum(self):
        WinCombo = [
            [[1, 2, 3], [4, 5, 6], [7, 8, 9], [1, 4, 7], [2, 5, 8], [3, 6, 9],
             [1, 5, 9], [3, 5, 7]],
            [[1, 2, 3], [4, 5, 6], [7, 8, 9], [1, 4, 7], [2, 5, 8], [3, 6, 9],
             [1, 5, 9], [3, 5, 7]],
            [[1, 2, 3], [4, 5, 6], [7, 8, 9], [1, 4, 7], [2, 5, 8], [3, 6, 9],
             [1, 5, 9], [3, 5, 7]],
            [[1, 2, 3], [4, 5, 6], [7, 8, 9], [1, 4, 7], [2, 5, 8], [3, 6, 9],
             [1, 5, 9], [3, 5, 7]],
            [[1, 2, 3], [4, 5, 6], [7, 8, 9], [1, 4, 7], [2, 5, 8], [3, 6, 9],
             [1, 5, 9], [3, 5, 7]],
            [[1, 2, 3], [4, 5, 6], [7, 8, 9], [1, 4, 7], [2, 5, 8], [3, 6, 9],
             [1, 5, 9], [3, 5, 7]],
            [[1, 2, 3], [4, 5, 6], [7, 8, 9], [1, 4, 7], [2, 5, 8], [3, 6, 9],
             [1, 5, 9], [3, 5, 7]],
            [[1, 2, 3], [4, 5, 6], [7, 8, 9], [1, 4, 7], [2, 5, 8], [3, 6, 9],
             [1, 5, 9], [3, 5, 7]],
            [[1, 2, 3], [4, 5, 6], [7, 8, 9], [1, 4, 7], [2, 5, 8], [3, 6, 9],
             [1, 5, 9], [3, 5, 7]]]
        count1 = 0
        for i in WinCombo:
            count = 0
            for k in i:
                self.bSum[count1][count] = 0
                for j in k:
                    self.bSum[count1][count] += self.bState[count1 + 1][j]
                count += 1
            count1 += 1
    def back_r_button(self):
        self.p1_button.grid_forget()
        self.p2_button.grid_forget()
        self.back_p_button.grid_forget()
        self.start_label.grid_forget()
        for a in range(1, 10):
            for b in range(1, 10):
                self.button_lst[a][b].grid_forget()
        self.StartGame()
class MainGUI(tkinter.Frame, pvp_app):
    def __init__(self, master=None):
        tkinter.Frame.__init__(self, master)
        self.grid()
        self.configure(background="green")
        self.master.title("Tic Tac Toe")
        self.master.geometry("500x530")
        self.ShowHomepage()
        self.label = tkinter.Label(self)
        self.label.grid(row=0, column=0)
    def ShowHomepage(self):
        self.homepage_label = tkinter.Label(self,
                                            text="WELCOME TO THE WORLD OF "
                                                 "TIC-TAC-TOE",
                                            font=("courier", 15), fg="red",
                                            bg="yellow")
        self.homepage_label.grid(row=0, column=1, padx=20, pady=25)
        self.start_button = tkinter.Button(self, text="Start Game",
                                           command=self.StartGame, fg="blue")
        self.start_button.grid(row=1, column=1, padx=100, pady=50)
        self.rules_button = tkinter.Button(self, text="Rules",
                                           command=self.ShowRule,
                                           fg="red")
        self.rules_button.grid(row=2, column=1, padx=100, pady=50)
        self.quit_button = tkinter.Button(self, text="Quit",
                                          command=self.QuitGame,
                                          fg="purple")
        self.quit_button.grid(row=3, column=1, padx=100, pady=50)
    def StartGame(self):
        self.rules_button.grid_forget()
        self.quit_button.grid_forget()
        self.homepage_label.grid_forget()
        self.start_button.grid_forget()
        self.startgame_label = tkinter.Label(self,
                                             text="Whom do you want to play against?",
                                             font=("courier", 15), fg="red",
                                             bg="yellow")
        self.startgame_label.grid(row=0, column=1, padx=20, pady=25)
        self.pvp_button = tkinter.Button(self, text="2 Player Mode",
                                         command=lambda: pvp_app.PlayerGame(
                                             self))
        self.pvp_button.grid(row=2, column=1, padx=100, pady=50)
        self.back_s_button = tkinter.Button(self, text="Back",
                                            command=self.back_startGame)
        self.back_s_button.grid(row=4, column=1, padx=100, pady=50)
    def ShowRule(self):
        self.homepage_label.grid_forget()
        self.start_button.grid_forget()
        self.rules_button.grid_forget()
        self.quit_button.grid_forget()
        self.rules_label = tkinter.Label(self,
                                         text="Game for two players,X and O,\nwho "
                                              "take turns marking the spaces\nin a "
                                              "3x3 grid.The player who\nsucceeds in "
                                              "placing three of their marks\nin a "
                                              "horizontal,vertical\nor diagonal row "
                                              "wins the game.",
                                         font=("pursia", 15), fg="red",
                                         bg="green")
        self.rules_label.grid(row=0, column=1, padx=45, pady=35)
        self.back_r_button = tkinter.Button(self, text="Back",
                                            command=self.back_rules)
        self.back_r_button.grid(row=2, column=1, padx=100, pady=50)
    def QuitGame(self):
        self.message = tkinter.messagebox.askquestion("Exit",
                                                      "Do you want to exit ?")
        if self.message == 'yes':
            appl.destroy()
    def back_rules(self):
        self.rules_label.grid_forget()
        self.back_r_button.grid_forget()
        self.ShowHomepage()
    def back_startGame(self):
        self.pvp_button.grid_forget()
        self.back_s_button.grid_forget()
        self.startgame_label.grid_forget()
        self.ShowHomepage()
appl = MainGUI()
appl.mainloop()
