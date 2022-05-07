from fileinput import FileInput
from xml.etree import ElementTree


CONFIG_FILE = "./GymTorcs/gym_torcs/raceconfigs/default.xml"


def get_track_name():
    """
    Parses the config XML to log out the current track name. Assumes there is only one track.

    Will include the category e.g. "road/g-track-1".
    """
    tree = ElementTree.parse(CONFIG_FILE)
    root = tree.getroot()
    track_name = ""
    category = ""

    for section in root:
        if section.attrib["name"] == "Tracks":
            for element in section:
                if element.tag == "section":
                    for entry in element:
                        if entry.attrib["name"] == "name":
                            track_name = entry.attrib["val"]
                        elif entry.attrib["name"] == "category":
                            category = entry.attrib["val"]

    return f"{category}/{track_name}"


def select_track(current, replacement):
    """
    Modifies the config XML to replace the 'existing' track with the 'new' one.

    Both should include the category e.g. "road/g-track-1".
    """
    [old_category, old_track_name] = str(current).split("/")
    [new_category, new_track_name] = str(replacement).split("/")

    replace_in_file(
        CONFIG_FILE,
        f"name=\"category\" val=\"{old_category}\"",
        f"name=\"category\" val=\"{new_category}\"")
    replace_in_file(
        CONFIG_FILE,
        f"name=\"name\" val=\"{old_track_name}\"",
        f"name=\"name\" val=\"{new_track_name}\"")


def replace_in_file(filename, find, replace):
    with FileInput(filename, inplace=True) as file:
        for line in file:
            print(line.replace(find, replace), end="")
