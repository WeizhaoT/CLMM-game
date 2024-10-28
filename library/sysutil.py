import os
import pandas as pd
import shutil

from typing import List, Dict, Union, Optional

F0 = 2**(1/12)

FREQ: Dict[str, float] = {
    'A': 1,
    'B': F0**2,
    'C': F0**3,
    'D': F0**5,
    'E': F0**7,
    'F': F0**8,
    'G': F0**10
}
""" Normalized frequency (based on "A") for each note. """


class LXC:
    " Collection of linux console colors. "
    RESET = "\033[0m"

    def red(string: str):
        """ Red """
        return "\033[31m" + string + LXC.RESET

    def red_lt(string: str):
        """ Light Red """
        return "\033[91m" + string + LXC.RESET

    def green_lt(string: str):
        """ Light Green """
        return "\033[92m" + string + LXC.RESET

    def yellow_lt(string: str):
        """ Light Yellow """
        return "\033[93m" + string + LXC.RESET

    def blue_lt(string: str):
        """ Light Blue """
        return "\033[94m" + string + LXC.RESET

    def cyan_lt(string: str):
        """ Light Cyan """
        return "\033[96m" + string + LXC.RESET


def sing(rhythm: Union[float, List[Union[float, List[float]]]],
         lengths: Union[float, List[float]] = 1.,
         base_length: float = .2,
         volume_dB: int = -10,
         extend: Optional[float] = None):
    """
    Generates and plays a sequence of musical notes based on the given rhythm and parameters.

    Args:
        rhythm (Union[float, List[Union[float, List[float]]]]): \
            The rhythm or sequence of notes to be played. \
            Can be \
                A) a single note, \
                B) a list of notes, or \
                C) a (nested) list of chords, where a chord is represented by a list of notes.
        lengths (Union[float, List[float]], optional): \
            The lengths of each note in the rhythm. Can be a single length or a list of lengths corresponding to each note. Defaults to 1.
        base_length (float, optional): \
            The base duration for each note length. Defaults to 0.2.
        volume_dB (int, optional): \
            The volume of the notes in decibels. Defaults to -10.
        extend (Optional[float], optional): \
            An optional extension factor for the last note's duration. If provided and non-negative, extends the last note by this factor.
    """
    if isinstance(rhythm, (int, float)):
        rhythm = [rhythm]
    if isinstance(lengths, (int, float)):
        lengths = [lengths] * len(rhythm)
    commands = []

    def parse_note(note: str) -> float:
        name, level = note[:2]
        mult = 1
        if len(note) == 3:
            mult = 2**(1/12) if note[2] == '#' else 2**(-1/12)
        return 440. * mult * (2 ** (int(level)-4)) * FREQ[name]

    last_len, last_notes = None, None
    for notes, length in zip(rhythm, lengths):
        command = []
        if isinstance(notes, str):
            notes = [notes]
        for note in notes:
            command.append(f'{length * base_length} sine {parse_note(note)}')

        commands.append("synth " + " ".join(command) + f" vol {int(volume_dB)}dB")
        last_len, last_notes = length * base_length, notes

    if isinstance(extend, (int, float)) and extend >= 0 and commands:
        commands[-1] = "synth " + \
            " ".join(f'{last_len * extend} sine {parse_note(note)}' for note in last_notes) + f" vol {int(volume_dB)}dB"
    os.system('play -nq ' + ' : '.join(commands))


def ensure_path(path: str, empty=False) -> None:
    """
    Ensures that a directory exists at the specified path. Optionally empties the directory if it already exists.

    Args:
        path (str): The path to the directory to ensure exists.
        empty (bool): If True, the directory will be emptied if it already exists. Defaults to False.

    Returns:
        None
    """
    if empty:
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path)
    elif not os.path.exists(path):
        os.makedirs(path)


def search_dir(pid: int) -> str:
    """
    Searches for a directory within the 'liquidity' folder that starts with the given process ID (pid). 
    If >= 1 exists, returns one path that matches the given process ID.

    Args:
        pid (int): The process ID to search for in the directory names.

    Returns:
        str: The path to the directory that matches the given process ID.

    Raises:
        FileNotFoundError: If no directory with the given process ID is found.
    """
    dirs = os.listdir('liquidity')
    found = False
    for d in dirs:
        if d.split('-')[0] == str(pid):
            found = True
            break

    if not found:
        raise FileNotFoundError(f'{pid} not found')

    return f'liquidity/{d}'


def get_info(pool_id: int) -> dict:
    """
    Retrieves information for a specific pool ID from a CSV file.

    Args:
        pid (int): Pool ID

    Returns:
        dict: A dictionary containing the information for the specified pool.
    """
    df = pd.read_csv('pairs.csv', index_col='pair_id')
    return df.loc[pool_id].to_dict()


def get_api_key() -> str:
    """
    Reads the API key from a file named 'API_KEY.txt' and returns it as a string.
    """
    with open('API_KEY.txt', 'r') as f:
        return f.read().strip()
