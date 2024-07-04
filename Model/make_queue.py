class pointer():

    def __init__(self, val):
        self.current = val
        self.next = None


class queue():

    def __init__(self):
        self.first = None


    def enqueue(self, val):
        point = pointer(val)

        if self.first == None:
            self.first = point

        elif self.first.next == None:
            self.first.next = point
            self.value = point

        else:
            self.value.next = point
            self.value = point

    def dequeue(self):
        current = self.first.current
        self.first = self.first.next
        return current