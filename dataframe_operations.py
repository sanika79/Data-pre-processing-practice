## important pandas dataframe operations
df.head(5)

df.isna()
df.dropna(inplace=True) 

df.drop
df.info()
df.describe
df.assign

5. Identify numerical and categorical columns
num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_cols = X.select_dtypes(include=["object"]).columns.tolist()

# for ordinal features
#Encode ordinal features with OrdinalEncoder
education_order = ['Basic', '2n Cycle','Graduation','Master', 'PhD']
oe = OrdinalEncoder(categories = [education_order], dtype=int)
education_oe = oe.fit_transform(df[['Education']])
df_enc= df.assign(Education_encode=education_oe)
print(df_enc.shape)
print(df_enc[['Education', 'Education_encode']])

# -----------------------------
# 3. Text Preprocessing
# -----------------------------
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = text.lower()
    ### Regular Expression Breakdown:
    # http\S+ → Matches http followed by any non-whitespace characters (\S+).
    # www\S+ → Matches www followed by non-whitespace characters.
    # https\S+ → Matches https followed by non-whitespace characters.
    # flags=re.MULTILINE → Ensures URLs appearing on multiple lines are removed.
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)  # Remove URLs

    # \d+ → Matches one or more digits (0-9).
    text = re.sub(r"\d+", "", text)  # Remove numbers

#     Regular Expression Breakdown:
# \w → Matches letters, digits, and underscores.
# \s → Matches whitespace (spaces, tabs, newlines).
# [^\w\s] → Matches anything that is NOT a word character or whitespace (i.e., punctuation).
    text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation
    text = " ".join([word for word in text.split() if word not in stop_words])  # Remove stopwords
    return text

df[text_column] = df[text_column].astype(str).apply(clean_text)




# # Encode nominal features with OneHotEncoder
# ohe =  OneHotEncoder(sparse=False, dtype='int')
# Marital_ohe = ohe.fit_transform(df[['Marital_Status']])



#filling the nan values generated
# scaled_df.isna().sum()
# scaled_df = scaled_df.fillna(0)
