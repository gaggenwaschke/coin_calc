import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterator, Tuple

from pandas import DataFrame, Index, Series, UInt64Dtype
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    value: int = Field(default=500, description="Value to create from given coins.")
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
LANGUAGES = [
    "ngerman"
]  # ["american-english", "british-english", "ngerman", "ogerman", "swiss"]


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
    valid_words = DataFrame(
        _get_all_valid_words(config, coin_values, letters_in_wallet, letter_index),
        columns=["word", "language", "value"],
    )
    valid_words.sort_values("value", ascending=True, inplace=True)
    print(valid_words)
    valid_words.to_csv(DATA_PATH / "output.csv")


def _get_all_valid_words(
    config: Config,
    coin_values: DataFrame,
    letters_in_wallet: Series,
    letter_index: Index,
) -> Iterator[Dict]:
    output_interval = timedelta(seconds=3)
    next_message = datetime.now() + output_interval
    word_count = sum(1 for _ in _get_all_words())
    base_series = Series(0, index=letter_index, dtype=UInt64Dtype())
    for index, language_and_word in enumerate(_get_all_words()):
        language, word = language_and_word

        now = datetime.now()
        if now > next_message:
            print("Analyzing {:.1f}%".format(100.0 * index / word_count))
            next_message = now + output_interval

        if len(word) < config.minimum_number_letters:
            continue
        word_letter_series = base_series.copy()
        for letter in _get_all_letters(word):
            word_letter_series.loc[letter] += 1

        # Skip the words we can not put together from wallet
        if (word_letter_series > letters_in_wallet).any():
            continue

        value = (coin_values.value * word_letter_series).sum()
        if value < config.value:
            continue

        yield dict(word=word, language=language, value=value)


def _get_all_letters(word: str) -> Iterator[str]:
    for letter in word:
        if letter == "Ä":
            yield "A"
            yield "E"
        elif letter == "Ö":
            yield "O"
            yield "E"
        elif letter == "Ü":
            yield "U"
            yield "E"
        elif letter == "É" or letter == "Ê":
            yield "E"
        elif letter == "Â":
            yield "A"
        elif letter == "'":
            continue
        else:
            yield letter


def _get_all_words() -> Iterator[Tuple[str, str]]:
    for language in LANGUAGES:
        with open(LANGUAGE_FILES_BASE_PATH / language, "rt") as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                yield language, line.upper()


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
