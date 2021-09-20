## WhatMe
The attached program mimics the chat pattern of a person using Natural Language Processing. A simple txt file of a chat between 2 people is fed into the program, and is then used as a training data for 2 seperate sub-programs namely, *Speak like Me* and *Respond Like Me*, making it easier for the user to let the bot do the talking while they can cater to something much more important. 

The program uses Tensorflow v1.15, SentencePiece and Google Sentence Encoder. The dataset is used my personal chats with one of my cousins. Since the chat I used is a small one, accuracy isn't that great but accuracy increases exponentially with the chat size.

Novelty:
1. Not everyone has the time and resources for chatting to consumers/people all time. This program makes it easier for the user to be available all the time.
2. Businesses can setup this bot to sound like the manager/owner and not some kind of trained ChatBot.
3. People can upload their chat patterns and the program can mimic them entirely, thus people can get a feeling of talking to their loved ones even after they're gone.  

Methodology:
1. Chats are exported to txt file and placed in the directory.
2. User and Other Person's names are changed as given in the txt file.
3. Chat file is then cleaned to remove unwanted text like: 'Missed Voice Call' or 'Image Omitted' <br>
`python clean_whatsapp_chats.py chats.txt`
4. The scratch file is then run and 2 functions namely 'respond-like-me' and 'talk-like-me' are created which output n number of lines which would sound like or respond like the user <br>
`python scratch.py`
5. The outputs are then shown on the sccreen.

Flowchart: <br><br>
![alt text](https://github.com/shreyagupta4/WhatME/blob/main/images/FLOWCHART.jpeg)

Input: <br><br>
![alt text](https://github.com/shreyagupta4/WhatME/blob/main/images/OUTPUT.jpeg)

Output: <br><br>
![alt text](https://github.com/shreyagupta4/WhatME/blob/main/images/INPUT.jpeg)
