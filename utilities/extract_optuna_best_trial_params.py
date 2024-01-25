import optuna
from optuna.storages import JournalStorage, JournalFileStorage
import argparse
import os


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-input", help="optuna journal file", required=True)
    parser.add_argument("-name", help="optuna study name", default=None)

    args = parser.parse_args()

    journal_file_path = args.input
    journal_file = os.path.basename(journal_file_path)
    default_study_name = os.path.splitext(journal_file)[0]

    study_name = args.name if args.name else default_study_name
    print("Loading study at : ", journal_file_path, "\twith name : ", study_name)

    storage = JournalStorage(JournalFileStorage(journal_file_path))


    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        load_if_exists=True
    )

    print("\n\tDisplaying parameters for best trial -- trial : ", study.best_trial.number)
    print("\n\n")
    for key, value in study.best_trial.params.items():
        print(key, "\t : ", value)
    
    print("\n\n")



