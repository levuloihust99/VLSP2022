import csv

file_writer = open("test_sampled_100.tsv", "w")
writer = csv.writer(file_writer, delimiter='\t')
with open("test.tsv", "r") as csvfile:
    reader = csv.reader(csvfile, delimiter='\t')
    count = 0
    for row in reader:
        writer.writerow(row)
        count += 1
        if count == 100:
            break
file_writer.close()
