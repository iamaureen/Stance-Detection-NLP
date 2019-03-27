import pandas as pd

#filename
filename = "data/emergent.csv"


fields = ['claim','claim_label','body','page_headline','page_position']

# #read entire csv file
# df = pd.read_csv(filename, header=None)
# print("df shape", df.shape) #(row, col)

# #read specific column
df = pd.read_csv(filename, usecols=fields)
print("df shape", df.shape)

#unique claim
claim_list = df.claim.unique()
print("total # of unique claims :: ",len(claim_list))
#
# #total count of each type of claim
# claim_count = df['claim'].value_counts();
# print(claim_count)

#unique headline
page_headline_list = df.page_headline.unique()
print("total # of unique headlines :: ",len(page_headline_list))

# #total count of each type of headline
# page_headline_count = df['page_headline'].value_counts();
# print(page_headline_count)

#unique stance
print("stance information")
page_position_list = df.page_position.unique()
print(page_position_list)

#total count of each type of stance
page_position_count = df['page_position'].value_counts();
print(page_position_count)

