import re
from pathlib import Path
from typing import Dict, Iterator, Tuple

from pandas import DataFrame, Index, Series, UInt64Dtype
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    value: int = Field(default=0, description="Value to create from given coins.")
    minimum_number_letters: int = Field(
        default=3, description="Minimum numbers of letter a word needs to have."
    )

    model_config = SettingsConfigDict(cli_parse_args=True)


DATA_PATH = Path(__file__).parent.parent / "data"
COIN_VALUE_PATH = DATA_PATH / "values.txt"
WALLET_PATH = DATA_PATH / "coins.txt"
COIN_VALUE_REGEX = re.compile(r"([a-zA-Z])\s(.*)\s([0-9]+)")
WALLET_LINE_REGEX = re.compile(r"([0-9]+)x([a-zA-Z ]+)([a-zA-Z])\s+\(([0-9]+)\)")
LANGUAGE_FILES_BASE_PATH = Path("/usr/share/dict/")
LANGUAGES = ["american-english", "british-english", "ngerman", "ogerman", "swiss"]


def main():
    config = Config()
    coin_values = DataFrame(_read_coin_values())
    coin_values.set_index("letter", inplace=True)
    letter_index = coin_values.index
    coins = DataFrame.from_dict(_read_coin_inventory(coin_values))
    value_all_coins = coins.value.sum()
    if value_all_coins < config.value:
        raise RuntimeError(
            "You do not have enough money (%d < %d).", value_all_coins, config.value
        )
    print(f"You wallet is worth {value_all_coins} Barr.")
    letters_in_wallet = Series(0, index=letter_index)
    letters_in_wallet += coins["letter"].value_counts()
    letters_in_wallet[letters_in_wallet.isna()] = 0
    letters_in_wallet = letters_in_wallet.astype(UInt64Dtype())
    valid_words = _get_all_valid_words(
        config, coin_values["value"], letters_in_wallet, letter_index
    )
    valid_words.sort_values("value", ascending=True, inplace=True)
    print(valid_words)
    valid_words[["language", "value"]].to_csv(DATA_PATH / "output.csv")


def _get_all_valid_words(
    config: Config,
    coin_values: Series,
    letters_in_wallet: Series,
    letter_index: Index,
) -> Iterator[Dict]:
    all_words = DataFrame.from_records(_get_all_words(), columns=["language", "word"])
    all_words.drop_duplicates("word", inplace=True)

    all_words = _extract_all_letters(all_words, letter_index)
    all_words.set_index("word", inplace=True)

    counts_dont_add_up = all_words.index.str.len() != all_words.drop(
        columns="language"
    ).sum(axis=1)
    if counts_dont_add_up.any():
        missmatched_words = all_words[counts_dont_add_up].index
        raise RuntimeError(
            f"Not all counts add up, you may need to handle more special characters!: {missmatched_words}"
        )

    # filter for number of letters
    all_words = all_words[all_words.index.str.len() >= config.minimum_number_letters]
    # filter for letters we need
    all_words = all_words[
        all_words.drop(columns=["language", "diff"]).le(letters_in_wallet).all(axis=1)
    ]
    # compute values
    all_words["value"] = (
        all_words.drop(columns="language").mul(coin_values, axis=1).sum(axis=1)
    )
    all_words = all_words[all_words["value"] >= config.value]
    all_words.sort_values("value", ascending=True, inplace=True)
    return all_words


def _extract_all_letters(all_words: DataFrame, letter_index: Index) -> DataFrame:
    all_words["diff"] = Series(all_words["word"].str.count("[']"))

    for letter in letter_index:
        regex = f"[{letter.lower()}{letter.upper()}]"
        all_words[letter] = Series(all_words["word"].str.count(regex))

    count = Series(all_words["word"].str.count("[äÄ]"))
    all_words["A"] += count
    all_words["E"] += count
    all_words["diff"] -= count

    count = Series(all_words["word"].str.count("[öÖ]"))
    all_words["O"] += count
    all_words["E"] += count
    all_words["diff"] -= count

    count = Series(all_words["word"].str.count("[üÜ]"))
    all_words["U"] += count
    all_words["E"] += count
    all_words["diff"] -= count

    count = Series(all_words["word"].str.count("[ß]"))
    all_words["S"] += 2 * count
    all_words["diff"] -= count

    count = Series(all_words["word"].str.count("[èéêÈÉÊ]"))
    all_words["E"] += count

    count = Series(all_words["word"].str.count("[âàáåÂÀÁÅ]"))
    all_words["A"] += count

    count = Series(all_words["word"].str.count("[ñÑ]"))
    all_words["N"] += count

    count = Series(all_words["word"].str.count("[íÍ]"))
    all_words["I"] += count

    count = Series(all_words["word"].str.count("[ôóÔÓ]"))
    all_words["O"] += count

    count = Series(all_words["word"].str.count("[çÇ]"))
    all_words["C"] += count

    count = Series(all_words["word"].str.count("[ûÛ]"))
    all_words["U"] += count

    return all_words


def _get_all_words() -> Iterator[Tuple[str, str]]:
    for language in LANGUAGES:
        with open(LANGUAGE_FILES_BASE_PATH / language, "rt") as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                yield language, line


def _read_coin_values():
    with open(COIN_VALUE_PATH, "rt", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            parameter = COIN_VALUE_REGEX.findall(line)[0]
            letter = parameter[0].upper()
            name = parameter[1]
            value = int(parameter[2])
            yield dict(letter=letter, name=name, value=value)


def _read_coin_inventory(coin_values: DataFrame):
    with open(WALLET_PATH, "rt", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            values = WALLET_LINE_REGEX.findall(line)[0]
            count = int(values[0])
            name = values[1].strip()
            letter = values[2].strip().upper()
            value = int(values[3]) / count
            expected_value = coin_values.loc[letter].value
            if value != expected_value:
                raise RuntimeError(
                    f"Your wallet contains values that do not match the coin value index!: Found {value}, but expected {expected_value}."
                )
            for _ in range(count):
                yield dict(name=name, letter=letter, value=value)


if __name__ == "__main__":
    main()
