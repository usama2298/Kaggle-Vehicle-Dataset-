import pandas as pd
from sklearn.preprocessing import StandardScaler, PolynomialFeatures


def StandScaler(df):
    col_names = df.columns.values.tolist()
    Scaler = StandardScaler()
    output_df = pd.DataFrame(Scaler.fit_transform(df))
    output_df.columns = col_names
    return output_df


def PolyFeatures(df, power):

    pf = PolynomialFeatures(power)
    output_nparray = pf.fit_transform(df)
    powers_nparray = pf.powers_

    col_names = list(df.columns)
    target_col_names = ["Constant Term"]
    for feature_distillation in powers_nparray[1:]:
        intermediary_label = ""
        final_label = ""
        for i in range(len(col_names)):
            if feature_distillation[i] == 0:
                continue
            else:
                variable = col_names[i]
                power = feature_distillation[i]
                intermediary_label = "%s^%d" % (variable, power)
                if final_label == "":
                    final_label = intermediary_label
                else:
                    final_label = final_label + " x " + intermediary_label
        target_col_names.append(final_label)
    output_df = pd.DataFrame(output_nparray, columns=target_col_names)
    return output_df
