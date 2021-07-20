import regex as re

def main():
    data_filepath = 'data/news-commentary-v15.en'
    female_attributes_filepath = 'data/female.txt'
    male_attributes_filepath = 'data/male.txt'
    stereotypes_filepath = 'data/stereotype.txt'

    with open(female_attributes_filepath) as f:
        female_attr = {l.strip() for l in f.readlines()}

    with open(male_attributes_filepath) as f:
        male_attr = {l.strip() for l in f.readlines()}

    with open(stereotypes_filepath) as f:
        stereo_attr = {l.strip() for l in f.readlines()}
    
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
                # print(f"L{iter+1}\t  MALE: ", male)
            
            if len(female) > 0 and len(male) == 0:
                female_cntr += 1
                # print(f"L{iter+1}\tFEMALE: ", female)
                
            if len(stereo) > 0 and len(male) == 0 and len(female) == 0:
                # print(f"L{iter+1}\t   STERE: ", stereo)
                stereo_cntr += 1

    print(f'male:   {male_cntr}')
    print(f'female: {female_cntr}')
    print(f'stereo: {stereo_cntr}')
    
    

if __name__ == "__main__":
    main()