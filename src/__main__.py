import re
from pathlib import Path
from typing import Dict, Iterator, Tuple

from pandas import DataFrame, Index, Series, UInt64Dtype
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    value: int = Field(default=300, description="Value to create from given coins.")
    value_range: int = Field(default=5, description="Upper value threshold")
    minimum_number_letters: int = Field(
        default=5, description="Minimum numbers of letter a word needs to have."
    )

    model_config = SettingsConfigDict(cli_parse_args=True)


DATA_PATH = Path(__file__).parent.parent / "data"
COIN_VALUE_PATH = DATA_PATH / "values.txt"
WALLET_PATH = DATA_PATH / "coins.txt"
COIN_VALUE_REGEX = re.compile(r"([a-zA-Z])\s(.*)\s([0-9]+)")
WALLET_LINE_REGEX = re.compile(r"([0-9]+)x([a-zA-Z ]+)([a-zA-Z])\s+\(([0-9]+)\)")
LANGUAGE_FILES_BASE_PATH = Path("/usr/share/dict/")
LANGUAGES = ["american-english", "british-english", "ngerman", "ogerman", "swiss"]

_KEY_WORD = "word"
_KEY_SANITIZED_WORD = "sanitized word"
_KEY_VALUE = "value"


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
    print(valid_words)
    valid_words.to_csv(DATA_PATH / "output.csv")


def _get_all_valid_words(
    config: Config,
    coin_values: Series,
    letters_in_wallet: Series,
    letter_index: Index,
) -> Iterator[Dict]:
    all_words = _get_all_words(config)
    coin_counts_per_word = _extract_all_letters(
        all_words[_KEY_SANITIZED_WORD], letter_index
    )
    all_words.drop(columns=_KEY_SANITIZED_WORD, inplace=True)
    word_values = coin_counts_per_word.mul(coin_values, axis=1).sum(axis=1)
    all_words[_KEY_VALUE] = word_values

    # filter for letters we need
    coin_count_filter = coin_counts_per_word.le(letters_in_wallet).all(axis=1)
    minimum_value_filter = all_words[_KEY_VALUE] >= config.value
    maximum_value_filter = all_words[_KEY_VALUE] <= (config.value + config.value_range)
    filtered_words = all_words[
        coin_count_filter & minimum_value_filter & maximum_value_filter
    ]
    # all_words = all_words[all_words[_KEY_VALUE] <= config.value + 10]
    filtered_words = filtered_words.sort_values(_KEY_VALUE, ascending=True)
    return filtered_words


def _extract_all_letters(all_sanitized_words: Series, letter_index: Index) -> DataFrame:
    letters = DataFrame(index=all_sanitized_words.index, columns=letter_index)
    diff = Series(0, index=all_sanitized_words.index)

    for letter in letter_index:
        regex = f"[{letter.lower()}{letter.upper()}]"
        letters[letter] = Series(all_sanitized_words.str.count(regex))

    count = Series(all_sanitized_words.str.count("[äÄ]"))
    letters["A"] += count
    letters["E"] += count
    diff -= count

    count = Series(all_sanitized_words.str.count("[öÖ]"))
    letters["O"] += count
    letters["E"] += count
    diff -= count

    count = Series(all_sanitized_words.str.count("[üÜ]"))
    letters["U"] += count
    letters["E"] += count
    diff -= count

    count = Series(all_sanitized_words.str.count("[ß]"))
    letters["S"] += 2 * count
    diff -= count

    count = Series(all_sanitized_words.str.count("[èéêÈÉÊ]"))
    letters["E"] += count

    count = Series(all_sanitized_words.str.count("[âàáåÂÀÁÅ]"))
    letters["A"] += count

    count = Series(all_sanitized_words.str.count("[ñÑ]"))
    letters["N"] += count

    count = Series(all_sanitized_words.str.count("[íÍ]"))
    letters["I"] += count

    count = Series(all_sanitized_words.str.count("[ôóÔÓ]"))
    letters["O"] += count

    count = Series(all_sanitized_words.str.count("[çÇ]"))
    letters["C"] += count

    count = Series(all_sanitized_words.str.count("[ûÛ]"))
    letters["U"] += count

    counts_dont_add_up = all_sanitized_words.str.len() != (letters.sum(axis=1) + diff)
    if counts_dont_add_up.any():
        missmatched_words = all_sanitized_words[counts_dont_add_up].index
        raise RuntimeError(
            f"Not all counts add up, you may need to handle more special characters!: {missmatched_words}"
        )

    return letters


def _get_all_words(config: Config) -> DataFrame:
    all_words = DataFrame.from_records(
        _iterate_all_language_files(), columns=["language", _KEY_WORD]
    )
    all_words.drop_duplicates(_KEY_WORD, inplace=True)
    all_words.set_index(_KEY_WORD, inplace=True)
    all_words[_KEY_SANITIZED_WORD] = all_words.index.str.replace("'", "")
    # filter for number of letters
    all_words = all_words[
        all_words[_KEY_SANITIZED_WORD].str.len() >= config.minimum_number_letters
    ]
    return all_words


def _iterate_all_language_files() -> Iterator[Tuple[str, str]]:
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
