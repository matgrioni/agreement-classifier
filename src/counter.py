class Counter(dict):
    def sum(self):
        return sum(self.values())

    def probability(self, key):
        return self[key] / self.sum()

    def __getitem__(self, key):
        try:
            # Must use this format because otherwise, there is a recursive
            # definition with no base case.
            return dict.__getitem__(self, key)
        except KeyError:
            return 0
