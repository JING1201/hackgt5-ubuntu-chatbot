import pandas

def read_conversations(file):
    conversations = [] #list of lists
    current_id = file['dialogueID'][0]
    current_conversation = []
    prev_user = ""
    for index, row in file.iterrows():
        if row['dialogueID'] != current_id:
            if len(current_conversation) > 1: #forming a conversation
                conversations.append(current_conversation)
            current_conversation = []
            prev_user = ""
            current_id = row['dialogueID']
        if row['from'] != prev_user:
            current_conversation.append(str(row['text']))
        else:
            current_conversation[len(current_conversation)-1] += str(row['text']) #one person talking continuously
        prev_user = row['from']
    return conversations

def get_tokenized_sequencial_sentences(conversations):
   for conversation in conversations:
       for i in range (len(conversation)-1):
           yield (conversation[i].split(" "), conversation[i+1].split(" "))
       '''
       max = len(conversation) - 1
       i = 0
       while i < max:
           yield (conversation[i].split(" "), conversation[i+1].split(" "))
           i += 2
        '''

    
def generate_conv_tuple(file):
    conversations = read_conversations(file)
    return tuple(zip(*list(get_tokenized_sequencial_sentences(conversations))))

def get_ubuntu_corpus_data():
    print("Getting ubuntu corpus data...")
    file = pandas.read_csv('Ubuntu-dialogue-corpus/dialogueText.csv') 
    #file = (pandas.read_csv('Ubuntu-dialogue-corpus/dialogueText_301.csv'))
    #file.append(pandas.read_csv('Ubuntu-dialogue-corpus/dialogueText_196.csv'))
    print("Finished getting ubuntu corpus data!")
    return generate_conv_tuple(file)
