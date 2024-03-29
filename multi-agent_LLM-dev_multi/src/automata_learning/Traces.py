import os
import functools

class Traces:
    def __init__(self, positive = set(), negative = set()):
        self.positive = positive
        self.negative = negative

    """
     IG: at the moment we are adding a trace only if it ends up in an event.
     should we be more restrictive, e.g. consider xxx, the same as xxxxxxxxxx (where x is an empty event '')
     recent suggestion (from the meeting): ignore empty events altogether and don't consider them as events at all (neither for
     execution, nor for learning)
     """
    def _should_add(self, trace, i):
        prefixTrace = trace[:i]
        if not prefixTrace[-1] == '':
            return True
        else:
            return False

    def _get_prefixes(self, trace, up_to_limit = None):
        if up_to_limit is None:
            up_to_limit = len(trace)
        all_prefixes = set()
        for i in range(1, up_to_limit+1):
            if self._should_add(trace, i):
                all_prefixes.add(trace[:i])
        return all_prefixes

    letters = ['a','b','d','e','f','g', 'h', 'n', 'l', 'c', 'p','j','k']
    # numbers = [int(i) for i in range(0,9)]
    numbers = [int(i) for i in range(0,14)]

    @classmethod
    @functools.lru_cache()
    def get_letter2number_dict(cls):
        return dict(zip(cls.letters, cls.numbers))

    @classmethod
    @functools.lru_cache()
    def get_number2letter_dict(cls):
        return dict(zip(cls.numbers, cls.letters))

    @classmethod
    def letter2number(cls, letter):
        return cls.get_letter2number_dict().get(letter, None)

    @classmethod
    def number2letter(cls, number):
        return cls.get_number2letter_dict().get(number, None)

    @classmethod
    def symbol_to_trace(cls, symbols):
        return tuple(cls.letter2number(letter) for letter in symbols)

    @classmethod
    def trace_to_symbol(cls, trace):
        return tuple(cls.number2letter(number) for number in trace)

    @classmethod
    def rm_trace_to_symbol(cls, rm_file):
        file = rm_file

        with open(file) as f:
            content = f.readlines()
        lines = []
        for line in content:
            end = 0
            begin = 1 #initialize values based on what won't enter the loops; initial values irrelevant
            number = 0 #random, had to initialize
            if line != content[0]:
                number = str()
                check = 0
                count=0
                for character in line:
                    if ((check==1) & (character=="'")): #looks for second quotation
                        check = 10 #end search
                        end = count-1
                    elif (character == "'"): #looks for first quotation
                        check = 1
                        begin = count+1
                    elif (check==1):
                        number += character
                    count = count+1
            symbol = cls.number2letter(int(number))
            #symbol = symbol + '&!n'
            line = list(line) #necessary for use of pop,insert
            if end==begin+1:
                line.pop(end)
                line.pop(begin)
                line.insert(begin,symbol)
            elif end==begin:
                line.pop(begin)
                line.insert(begin,symbol)
            lines.append(line)
        with open(rm_file, 'w') as f:
            for line in lines:
                for item in line:
                    f.write(str(item))

    @staticmethod
    def fix_rmfiles(rmfile):
        file = rmfile
        with open(file) as f:
            content = f.readlines()

        final_state = str()

        for line in content:
            if line != content[0]:
                brackets = 0
                commas = 0
                state = str()
                next_state = str()
                for character in line:
                    if (character == "(") & (brackets == 0):
                        brackets = 1
                    elif brackets == 1:
                        if character == "(":
                            brackets = 2
                    elif brackets == 2:
                        if character == "1":
                            final_state = next_state
                            print(final_state)
                    if ((commas == 0) & (brackets == 1)):
                        if character == ",":
                            commas = 1
                        else:
                            state += character
                    elif ((commas == 1) & (brackets == 1)):
                        if character == ",":
                            commas = 2
                        else:
                            next_state += character

        # with open(rmfile, 'w') as f:
        #     for line in content:
        #         for item in line:
        #             f.write(str(item))
        #     f.write("\n")
        #     writethis = "(" + str(final_state) + "," + str(final_state) + ",'True',ConstantRewardFunction(0))"
        #     f.write(writethis)
    """
    when adding a trace, it additionally adds all prefixes as negative traces
    """
    def add_trace(self, trace, reward, learned):
        trace = tuple(trace)
        print('trace in Traces class:', trace)
        if reward > 0:
            self.positive.add(trace)
            # | is a set union operator
            #if learned==0:
            self.negative |= self._get_prefixes(trace, len(trace)-1)

        else:
            #if learned == 0:
            self.negative |= self._get_prefixes(trace)
            # else:
            #   self.negative.add(trace)

    def export_traces(self, filename):
        parent_path = os.path.dirname(filename)
        os.makedirs(parent_path,exist_ok=True)
        with open(filename, "w") as output_file:
            output_file.write("POSITIVE:")
            for trace in self.positive:
                output_file.write("\n")
                string_repr = [str(el) for el in trace]
                output_file.write(','.join(string_repr))
            output_file.write("\nNEGATIVE:")
            for trace in self.negative:
                output_file.write("\n")
                string_repr = [str(el) for el in trace]
                output_file.write(','.join(string_repr))


    def __repr__(self):
        return repr(self.positive) + "\n\n" + repr(self.negative)
