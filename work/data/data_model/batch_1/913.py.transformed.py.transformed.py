class BaseReporter(object):
    def starting(self):
    def starting_round(self, index):
    def ending_round(self, index, state):
    def ending(self, state):
    def adding_requirement(self, requirement, parent):
    def backtracking(self, candidate):
    def pinning(self, candidate):
