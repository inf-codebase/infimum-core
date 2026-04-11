import re
from datetime import datetime, timedelta
import glob


def get_time_in_string(format = None, back_to_n_days = 0):
    if format is None:
        format = "%Y%m%d_%H%M%S"
    current_time_in_string = datetime.today() - timedelta(days=back_to_n_days)
    current_time_in_string = current_time_in_string.strftime(format)
    return current_time_in_string

def get_all_file_paths_from_folder(folder):
    file_paths = glob.glob(f"{folder}/*")
    file_paths.sort()
    return file_paths

def camel_to_plural_underscore(class_name):
    """
    Convert a camel case class name to an underscored plural form.

    Examples:
        PersonAddress -> person_addresses
        Box -> boxes
        Category -> categories
        Match -> matches
        Story -> stories
    """
    # Handle empty input
    if not class_name:
        return ""

    # Insert underscore between camel case words
    name = re.sub(r'(?<!^)(?=[A-Z])', '_', class_name).lower()

    # Basic pluralization rules
    if name.endswith('y'):
        # category -> categories
        name = name[:-1] + 'ies'
    elif name.endswith(('s', 'x', 'z', 'ch', 'sh')):
        # box -> boxes, match -> matches
        name = name + 'es'
    else:
        # person -> persons
        name = name + 's'

    return name
