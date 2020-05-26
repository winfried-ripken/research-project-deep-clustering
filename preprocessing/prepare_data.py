import csv
import numpy as np

def main():
    data = []

    with open('data/filtered_commentaires.csv', 'r', encoding="iso8859_16") as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')

        for row in reader:
            row = row[3]

            if len(row) > 1:
                data.append(row)

    print("number of statements: " + str(len(data)))
    training_test_split = int(len(data) * 0.9)

    data = np.array(data)
    np.random.shuffle(data)

    train = data[:training_test_split]
    test = data[training_test_split:]

    print("number in train: " + str(len(train)))
    print("number in test: " + str(len(test)))

    np.save("data/train", train)
    np.save("data/test", test)

if __name__ == "__main__":
    main()