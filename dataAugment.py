def crop(df, key, start):
    startIdx = df.loc[df[key] == start].index[0]
    endIdx = df.loc[df[key] == start+1].index[0]
    df = df.loc[startIdx:endIdx-1] 
    return df

def scale(df,key, value):
    df[key] = df[key]*value
    df[key] = (round(df[key]))
    return df

def mask(df, key, value):
    return df[df[key] == value]