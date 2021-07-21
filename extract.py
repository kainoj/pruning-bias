import regex as re

def get_attribute_set(filepath: str) -> set:
    """Reads file with attributes and returns a set containing them all"""
    with open(filepath) as f:
        return {l.strip() for l in f.readlines()}


def main():
    data_filepath = 'data/news-commentary-v15.en'
    female_attributes_filepath = 'data/female.txt'
    male_attributes_filepath = 'data/male.txt'
    stereotypes_filepath = 'data/stereotype.txt'

    female_attr = get_attribute_set(female_attributes_filepath)
    male_attr = get_attribute_set(male_attributes_filepath)
    stereo_attr = get_attribute_set(stereotypes_filepath)
    
    # This regexp basically tokenizes a sentence over spaces and 's, 're, 've..
    # It's originally taken from OpenAI's GPT-2 Encoder implementation
    pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

    male_cntr = 0
    female_cntr = 0
    stereo_cntr = 0

    with open(data_filepath) as f:
        for iter, full_line in enumerate(f.readlines()):

            line = full_line.strip()

            if len(line) < 1 or len(line.split()) > 128 or len(line.split()) <= 1:
                continue

            line_tokenized = {token.strip().lower() for token in re.findall(pat, line)}
            
            male = line_tokenized & male_attr
            female = line_tokenized & female_attr
            stereo = line_tokenized & stereo_attr

            if len(male) > 0 and len(female) == 0:
                male_cntr += 1
            
            if len(female) > 0 and len(male) == 0:
                female_cntr += 1
                
            if len(stereo) > 0 and len(male) == 0 and len(female) == 0:
                stereo_cntr += 1


if __name__ == "__main__":
    main()