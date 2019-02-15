
# coding: utf-8

# In[7]:


import json


# In[8]:


sc_sc = open("sc_sc.txt", "w")  # subject+content \t subject+content
c_s = open("c_s.txt", "w")  # content \t subject
sa_sa = open("sa_sa.txt", "w")  # subject+answer \t subject+answer 


# In[9]:


"""
    Reading in the data and splitting it into the training pairs
"""

with open("./data/yahooAnswers.json") as f:
    data = json.load(f)
    items = 0  # count total examples
    for s in data:
        if len(s["subject"]) > 0 and len(s["content"]) > 0 and len(s["chosenanswer"]) > 0:
            items+= 1

            sc = " ".join(s["subject"].splitlines()).strip() + " " + " ".join(s["content"].splitlines()).strip()
            sa = " ".join(s["subject"].splitlines()).strip() + " " + " ".join(s["chosenanswer"].splitlines()).strip()
            
            sc_sc.write("%s\t%s\n" % (sc, sc))
            c_s.write("%s\t%s\n" %
                      (" ".join(s["content"].splitlines()), " ".join(s["subject"].splitlines())))
            sa_sa.write("%s\t%s\n" % (sa, sa))

            
print("Done. Lines: ", items)


# In[10]:


# Sanity checks, making sure the number of lines is equal among the files
sc_read = open("sc_sc.txt", "r")
cs_read = open("c_s.txt", "r")
sa_read = open("sa_sa.txt", "r")

i = 0
for l in sc_read:
    i+= 1
print(i)
i = 0
for l in cs_read:
    i+= 1
print(i)
i = 0
for l in sa_read:
    i+= 1
print(i)


# In[11]:


sa_read.close()


# In[12]:


sc_read.close()
cs_read.close()


# In[13]:


sc_sc.close()
c_s.close()
sa_sa.close()

