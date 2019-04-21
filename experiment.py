import pandas as pd
import sys
import random

"""
First question, which takes up more memory? Panda data frame vs loaded dictionary?
Well, frame memory usage appears to be lower in terms of raw data, but Panda's abstracts 
it frame, so getting the actual size uses getsizeof from the sys module.
For size of dictionary, using getsizeof works if it is applied to each element.


dataTable_with_data_frame = pd.read_csv('example.csv')
print(dataTable_with_data_frame.memory_usage(index=True).sum())
print(sys.getsizeof(dataTable_with_data_frame))

data_without_data_frame = {"BusNumber": [51, "F"], "Location": ["Berkeley", "Berkeley"],
                           "Start_Road": ["Cory Hall", "2551 Hearst Ave"], "End_Road": ["Gayley Road", "Gayley Road"],
                           "Time": [1, 3]}

def deepSizeOf(dictionaryData):
    total = sys.getsizeof(dictionaryData)
    for eachList in dictionaryData.values():
        total += sys.getsizeof(eachList)
    return total


print(deepSizeOf(data_without_data_frame),"\n")
"""

"""
Second, the project. Time is in minutes. Distance measured by "sensor" is my estimate using a finger. 
Using actransit's data about a bus around Berkeley (measured real time for the location),
I tracked line 52 (3 buses) with a few samples and added them as prior samples.
"""

priorSamples = [["miles", 0.2, 1400, 1, "green"], ["miles", 0.3, 1600, 3, "orange"], ["feet", 1000, 1200, 3.36, "red"],
                ["feet", 820, 850, 42 / 60, "orange"], ["feet", 1672, 1400, 1, "green"],
                ["feet", 1537, 1500, 2, "orange"], ["feet", 2312, 2300, 8.3, "maroon"],
                ["feet", 2289, 2300, 3.6, "green"], ["feet", 3563, 3520, 3 + 17 / 60, "orange"],
                ["feet", 954, 1000, 1.1, "green"], ["feet", 954, 700, 43 / 60, "green"],
                ["feet", 1431, 1200, 1.5, "green"]]

test_data = [["miles", 0.2, 1400, 0.84, "green"], ["miles", 0.56, 2200, 3, "red"], ["feet", 1935, 1700, 1, "green"],
             ["miles", 1.1, 5100, 2.5, "green"], ["miles", 0.2, 1400, 2, "maroon"],
             ["feet", 1537, 1500, 2.4, "green"], ["feet", 5112, 5000, 3.8, "green"],
             ["feet", 1298, 1000, 2.9, "red"], ["feet", 3563, 3520, 3 + 17 / 60, "orange"],
             ["feet", 954, 1000, 0.8, "green"], ["feet", 954, 700, 0.5, "green"],
             ["feet", 1859, 1600, 1.8, "green"], ["miles", 1.11, 5500, 2.4, "green"],
             ["feet", 2300, 2500, 1.1, "orange"], ["feet", 3718, 3700, 1.4, "green"]]


# this simulator generates very fake samples as opposed to the samples at the top which are real life hand picked
def simulationFunction(numSamples):
    fake_data = []
    possible = {0: "green", 1: "orange", 2: "red", 3: "maroon"}
    for i in range(numSamples):
        supposedFeet = random.random() * 5000
        supposedSensorFeet = supposedFeet + random.randint(-200, 200)

        fake_data.append(
            ["feet", supposedFeet, supposedSensorFeet, random.random() * 4 + 1, possible[int(random.random() * 4)]])

    return fake_data


class Distribution(dict):

    def __getitem__(self, key):
        self.setdefault(key, 0)
        return dict.__getitem__(self, key)

    def __init__(self, orderOfElements=(), parents=None):
        self.domain = tuple(orderOfElements)
        self.parents = parents  # for conditionals

    def copy(self):
        return Distribution(dict.copy(self))

    def total(self):
        return float(sum(self.values()))

    def normalize(self):

        total = self.total()
        newNormalized = Distribution()
        if (total == 0):
            return
        for key in self.keys():
            newNormalized[key] = self[key] / total
        return newNormalized


# OPTIONAL SINCE VARIABLE ELIMINATION CAN HELP WITH ENOUGH SAMPLED DATA########
dayDistribution = Distribution()


def computeDayDistribution():
    equalProbability = 0.14285714285
    dayDistribution["Monday"] = equalProbability
    dayDistribution["Tuesday"] = equalProbability
    dayDistribution["Wednesday"] = equalProbability
    dayDistribution["Thursday"] = equalProbability
    dayDistribution["Friday"] = equalProbability
    dayDistribution["Saturday"] = equalProbability
    dayDistribution["Sunday"] = equalProbability


computeDayDistribution()
##############################################################


sensorDistribution = Distribution()  # reliable along consistent distances
trafficColorDistribution = Distribution()  # unreliable at low number samples
timeDistribution = Distribution()  # reliable along consistant time ranges
velocityDistribution = Distribution()


# for calibration of visually seeing the sample
def milesToFeet(miles):
    return miles * 5280


def feetToMiles(feet):
    return feet / 5280


def sensorConversionSample(fingerDistance, trueDistance, feetOrMiles):
    if (feetOrMiles != "feet"):
        diff = fingerDistance - milesToFeet(trueDistance)
    else:
        diff = fingerDistance - trueDistance
    sensorDistribution[roundToNearestHundred(diff)] += 1


# only call this when sensorDistribution has enough samples
def expectedVelocity(sensorDistance, time):
    velocity = 0
    for pair in sensorDistribution.normalize().items():
        velocity += (pair[0] + sensorDistance) / time * pair[1]
    return feetToMiles(velocity) * 60


# keep the velocity within a likely bus velocity (note this is not accurate, since measuring time itself has error)
cachedQueries = {}


def inferTimeFromVelocity(time, velocity, predictor):
    normalizedVelocity = velocityDistribution.normalize()

    probabilityPredictorAndVelocity = 0
    cached = False
    if ((velocity, predictor) in cachedQueries.keys()):
        cached = True
        probabilityPredictorAndVelocity = cachedQueries[(velocity, predictor)]

    jointNumerator = 0
    for jointQuery in normalizedVelocity:
        if (not cached and jointQuery[1] == predictor and
                roundToNearestFive(jointQuery[2]) == roundToNearestFive(velocity)):
            probabilityPredictorAndVelocity += normalizedVelocity[jointQuery]
        if (jointQuery[1] == predictor and roundToNearestFive(jointQuery[2]) == roundToNearestFive(velocity)
                and jointQuery[0] == roundToNearestHalf(time)):
            jointNumerator += normalizedVelocity[jointQuery]

    return jointNumerator / probabilityPredictorAndVelocity


def roundToNearestHalf(num):
    return round(num * 2) / 2


def roundToNearestHundred(num):
    return round(num) // 10 * 10


def roundToNearestFive(num):
    return 5 * round(num / 5)


def load():
    for sample in priorSamples:
        sensorConversionSample(sample[2], sample[1], sample[0])
        timeDistribution[roundToNearestHalf(sample[3])] += 1
        trafficColorDistribution[sample[4]] += 1


def testFunctions():
    print(timeDistribution.normalize())
    print(trafficColorDistribution.normalize())
    print(sensorDistribution.normalize())
    print(velocityDistribution)
    print("100mph expected in this distribution is", expectedVelocity(milesToFeet(100), 60))

    print(expectedVelocity(1000, 1))


load()


def test():
    for sample in test_data:
        sensorConversionSample(sample[2], sample[1], sample[0])
        timeRounded = roundToNearestHalf(sample[3])
        timeDistribution[timeRounded] += 1
        trafficColorDistribution[sample[4]] += 1
        key = (timeRounded, sample[4], expectedVelocity(sample[2], timeRounded))
        velocityDistribution[key] += 1


# bad samples but used for convenience of testing large data simulation
def test2():
    fake = simulationFunction(100)
    for sample in fake:
        sensorConversionSample(sample[2], sample[1], sample[0])
        timeRounded = roundToNearestHalf(sample[3])
        timeDistribution[timeRounded] += 1
        trafficColorDistribution[sample[4]] += 1
        key = (timeRounded, sample[4], expectedVelocity(sample[2], timeRounded))
        velocityDistribution[key] += 1


test()
test2()  # comment, for true demonstration, but it won't be accurate since these are bad unrealistic samples
testFunctions()
accuracyPredictor = 0
for i in range(1, 15):
    inference = inferTimeFromVelocity(i * 0.5, 10, "green")
    accuracyPredictor += inference
    print("The probability that time is", i * 0.5, "min given 10 mph velocity and green predictor is", inference)
print("This should be near one:", accuracyPredictor)
