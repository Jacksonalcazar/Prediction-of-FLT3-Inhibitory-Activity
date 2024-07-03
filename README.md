# Prediction-of-FLT3-Inhibitory-Activity
This GitHub repository provides a script-based tool aimed at enhancing user experience by automating the prediction of pIC<sub>50</sub> and IC<sub>50</sub> values for any compound using its SMILES code. The model's reliability depends on the structural similarity of the target compound to the structures involved in its training. Designed to be user-friendly, this tool delivers results swiftly, within seconds."

## Prerequisites

Before running the tool, ensure you have Python installed on your system. You can download Python [here](https://www.python.org/downloads/).

Additionally, you will need the following libraries:
- pandas
- joblib

You can install these libraries using the following command:

pip install pandas joblib

## Getting Started

To use this tool, follow these simple steps:

1. Clone this repository to your local machine:

git clone https://github.com/Jacksonalcazar/Prediction-of-FLT3-Inhibitory-Activity.git

Or simply download the files to your device.

2. Run the `RUN.bat` file.

RUN.bat

## Classification Criteria

The tool processes compound data by classifying their activity levels based on pIC<sub>50</sub> values and categorizing the reliability of predictions according to the similarity scores. Below are the steps and criteria used for these classifications:

### Activity Classification

Activity levels are determined using the pIC<sub>50</sub> values as follows:

- **High Activity**: Compounds with ab pIC<sub>50</sub> value of 8 or greater.
- **Medium Activity**: Compounds with a pIC<sub>50</sub> value greater than 6 and up to 8.
- **Low Activity**: Compounds with a pIC<sub>50</sub> value lower than 6.

### Similarity and Reliability Categorization

The similarity percentage is then used to categorize the reliability of the predictions:

- **High Reliability**: Compounds with a similarity score of 80% or higher.
- **Medium Reliability**: Compounds with a similarity score between 65% and 79%.
- **Low Reliability**: Compounds with a similarity score below 65%.

## Support

For any issues or questions, please open an issue in this repository, and we'll get back to you as soon as possible.
