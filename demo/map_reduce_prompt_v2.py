from langchain.chains.prompt_selector import ConditionalPromptSelector, is_chat_model
from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
    MessagesPlaceholder
)
from langchain_core.prompts.prompt import PromptTemplate
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
question_prompt_template = """Use the following portion of a long document to see if any of the text is relevant to answer the question. 
Return any relevant text verbatim.
{context}
Question: {question}
Relevant text, if any:"""
QUESTION_PROMPT = PromptTemplate(
    template=question_prompt_template, input_variables=["context", "question"]
)
system_template = """Use the following portion of a long document to see if any of the text is relevant to answer the question. 
Return any relevant text verbatim.
_______________________
{context}"""
messages = [
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template("{question}"),
]
CHAT_QUESTION_PROMPT = ChatPromptTemplate.from_messages(messages)


QUESTION_PROMPT_SELECTOR = ConditionalPromptSelector(
    default_prompt=QUESTION_PROMPT, conditionals=[(is_chat_model, CHAT_QUESTION_PROMPT)]
)

combine_prompt_template = """Given the following extracted parts of a long document and a question, create a final answer. 
If you don't know the answer, just say that you don't know. Don't try to make up an answer.

QUESTION: Which state/country's law governs the interpretation of the contract?
=========
Content: This Agreement is governed by English law and the parties submit to the exclusive jurisdiction of the English courts in  relation to any dispute (contractual or non-contractual) concerning this Agreement save that either party may apply to any court for an  injunction or other relief to protect its Intellectual Property Rights.

Content: No Waiver. Failure or delay in exercising any right or remedy under this Agreement shall not constitute a waiver of such (or any other)  right or remedy.\n\n11.7 Severability. The invalidity, illegality or unenforceability of any term (or part of a term) of this Agreement shall not affect the continuation  in force of the remainder of the term (if any) and this Agreement.\n\n11.8 No Agency. Except as expressly stated otherwise, nothing in this Agreement shall create an agency, partnership or joint venture of any  kind between the parties.\n\n11.9 No Third-Party Beneficiaries.

Content: (b) if Google believes, in good faith, that the Distributor has violated or caused Google to violate any Anti-Bribery Laws (as  defined in Clause 8.5) or that such a violation is reasonably likely to occur,
=========
FINAL ANSWER: This Agreement is governed by English law.

QUESTION: What did the president say about Michael Jackson?
=========
Content: Madam Speaker, Madam Vice President, our First Lady and Second Gentleman. Members of Congress and the Cabinet. Justices of the Supreme Court. My fellow Americans.  \n\nLast year COVID-19 kept us apart. This year we are finally together again. \n\nTonight, we meet as Democrats Republicans and Independents. But most importantly as Americans. \n\nWith a duty to one another to the American people to the Constitution. \n\nAnd with an unwavering resolve that freedom will always triumph over tyranny. \n\nSix days ago, Russia’s Vladimir Putin sought to shake the foundations of the free world thinking he could make it bend to his menacing ways. But he badly miscalculated. \n\nHe thought he could roll into Ukraine and the world would roll over. Instead he met a wall of strength he never imagined. \n\nHe met the Ukrainian people. \n\nFrom President Zelenskyy to every Ukrainian, their fearlessness, their courage, their determination, inspires the world. \n\nGroups of citizens blocking tanks with their bodies. Everyone from students to retirees teachers turned soldiers defending their homeland.

Content: And we won’t stop. \n\nWe have lost so much to COVID-19. Time with one another. And worst of all, so much loss of life. \n\nLet’s use this moment to reset. Let’s stop looking at COVID-19 as a partisan dividing line and see it for what it is: A God-awful disease.  \n\nLet’s stop seeing each other as enemies, and start seeing each other for who we really are: Fellow Americans.  \n\nWe can’t change how divided we’ve been. But we can change how we move forward—on COVID-19 and other issues we must face together. \n\nI recently visited the New York City Police Department days after the funerals of Officer Wilbert Mora and his partner, Officer Jason Rivera. \n\nThey were responding to a 9-1-1 call when a man shot and killed them with a stolen gun. \n\nOfficer Mora was 27 years old. \n\nOfficer Rivera was 22. \n\nBoth Dominican Americans who’d grown up on the same streets they later chose to patrol as police officers. \n\nI spoke with their families and told them that we are forever in debt for their sacrifice, and we will carry on their mission to restore the trust and safety every community deserves.

Content: And a proud Ukrainian people, who have known 30 years  of independence, have repeatedly shown that they will not tolerate anyone who tries to take their country backwards.  \n\nTo all Americans, I will be honest with you, as I’ve always promised. A Russian dictator, invading a foreign country, has costs around the world. \n\nAnd I’m taking robust action to make sure the pain of our sanctions  is targeted at Russia’s economy. And I will use every tool at our disposal to protect American businesses and consumers. \n\nTonight, I can announce that the United States has worked with 30 other countries to release 60 Million barrels of oil from reserves around the world.  \n\nAmerica will lead that effort, releasing 30 Million barrels from our own Strategic Petroleum Reserve. And we stand ready to do more if necessary, unified with our allies.  \n\nThese steps will help blunt gas prices here at home. And I know the news about what’s happening can seem alarming. \n\nBut I want you to know that we are going to be okay.

Content: More support for patients and families. \n\nTo get there, I call on Congress to fund ARPA-H, the Advanced Research Projects Agency for Health. \n\nIt’s based on DARPA—the Defense Department project that led to the Internet, GPS, and so much more.  \n\nARPA-H will have a singular purpose—to drive breakthroughs in cancer, Alzheimer’s, diabetes, and more. \n\nA unity agenda for the nation. \n\nWe can do this. \n\nMy fellow Americans—tonight , we have gathered in a sacred space—the citadel of our democracy. \n\nIn this Capitol, generation after generation, Americans have debated great questions amid great strife, and have done great things. \n\nWe have fought for freedom, expanded liberty, defeated totalitarianism and terror. \n\nAnd built the strongest, freest, and most prosperous nation the world has ever known. \n\nNow is the hour. \n\nOur moment of responsibility. \n\nOur test of resolve and conscience, of history itself. \n\nIt is in this moment that our character is formed. Our purpose is found. Our future is forged. \n\nWell I know this nation.
=========
FINAL ANSWER: The president did not mention Michael Jackson.

QUESTION:  介紹數位發展部主管財團法人預決算與會計處理及財務報告編製準則第二章
=========
Content: 根據提供的上下文，無法找到有關數位發展部主管財團法人預決算與會計處理及財務報告編製準則的第二章的相關資訊。建議查閱正式的法規文件或相關資料以獲取更詳細的信息。

Content: 根據提供的上下文，數位發展部主管財團法人的預算與會計處理以及財務報告編製準則第二章可能包括以下內容：
            1. 預算編制：包括財團法人的年度預算編制流程、預算編制原則、預算編制的財務指標和目標等。
            2. 會計處理：涉及財團法人的會計制度設置、會計政策、會計記錄和報告程序、會計核算方法等。
            3. 財務報告編製準則：包括財團法人的財務報告編製流程、財務報告的內容和格式、財務報告的披露要求、財務報告的審核和核准程序等。

            這些準則旨在確保財團法人的財務管理和報告符合相關法規要求，並提供透明度和準確性，以便監督機關和利害關係人能夠了解財團法人的財務狀況和營運表現。詳細的內容可能需要參考具體的法規文件或相關指引。

Content: 根據提供的上下文，無法找到有關數位發展部主管財團法人預決算與會計處理及財務報告編製準則第二章的相關資訊。建議查閱數位發展部相關的法規文件或官方網站，以獲取更詳細的資訊。

=========
FINAL ANSWER: 根據提供的上下文，數位發展部主管財團法人的預算與會計處理以及財務報告編製準則第二章可能包括以下內容：
            1. 預算編制：包括財團法人的年度預算編制流程、預算編制原則、預算編制的財務指標和目標等。
            2. 會計處理：涉及財團法人的會計制度設置、會計政策、會計記錄和報告程序、會計核算方法等。
            3. 財務報告編製準則：包括財團法人的財務報告編製流程、財務報告的內容和格式、財務報告的披露要求、財務報告的審核和核准程序等。

            這些準則旨在確保財團法人的財務管理和報告符合相關法規要求，並提供透明度和準確性，以便監督機關和利害關係人能夠了解財團法人的財務狀況和營運表現。詳細的內容可能需要參考具體的法規文件或相關指引。

QUESTION: {question}
=========
Content: {summaries}
=========
FINAL ANSWER:"""
COMBINE_PROMPT = PromptTemplate(
    template=combine_prompt_template, input_variables=["summaries", "question"]
)

system_template = """
根據提供的資訊，盡可能詳細地回答問題，確保提供所有細節。如果在文中有出現相關的資訊請將這些資訊整理成大綱。如果段落中出現"沒有提供"或"不清楚"或"無法回答"或"無法找到"以及類似否定資訊，請無視這些否定段落，並使用有用的段落回答。
    例子:
        question: 介紹數位發展部主管財團法人預決算與會計處理及財務報告編製準則第二章
        
        summeries: 
            根據提供的上下文，無法找到有關數位發展部主管財團法人預決算與會計處理及財務報告編製準則的第二章的相關信息。該文件可能包含有關財團法人的預算編製、會計處理、財務報告編製等方面的準則和規定。建議查閱原始文件以獲取更詳細的信息。

            根據提供的上下文，數位發展部主管財團法人的預決算與會計處理以及財務報告編製準則可能包括以下內容：

            1. 預算編制：財團法人應根據其業務計畫和目標，制定年度預算，並在特定時間前提交給相關機構進行核准。

            2. 會計處理：財團法人應遵守相關的會計準則和法規，確保財務交易的準確記錄和處理。這包括記錄收入、支出、資產和負債等信息。

            3. 財務報告編製：財團法人應根據法律要求和會計準則，編製年度財務報告。這些報告應包括資產負債表、損益表和現金流量表等財務信息。

            4. 內部控制：財團法人應建立有效的內部控制制度，以確保財務交易的合規性和準確性。這包括風險管理、審計和稽核等措施。

            5. 績效評估：財團法人應定期進行績效評估，以評估其業務目標的達成情況，並根據評估結果調整相應的策略和計劃。

            總的來說，數位發展部主管財團法人的預決算與會計處理及財務報告編製準則第二章將涵蓋財務管理的各個方面，以確保財團法人的財務運作合規且有效。詳細的內容可能需要參考具體的法規文件或相關指引。

            根據提供的上下文，無法找到有關數位發展部主管財團法人預決算與會計處理及財務報告編製準則第二章的相關資訊。因此，無法提供有關該主題的介紹。建議查閱數位發展部相關的法規文件或官方網站以獲取更多資訊。
        
        answer: 根據提供的上下文，數位發展部主管財團法人的預決算與會計處理以及財務報告編製準則可能包括以下內容：

                1. 預算編制：財團法人應根據其業務計畫和目標，制定年度預算，並在特定時間前提交給相關機構進行核准。

                2. 會計處理：財團法人應遵守相關的會計準則和法規，確保財務交易的準確記錄和處理。這包括記錄收入、支出、資產和負債等信息。

                3. 財務報告編製：財團法人應根據法律要求和會計準則，編製年度財務報告。這些報告應包括資產負債表、損益表和現金流量表等財務信息。

                4. 內部控制：財團法人應建立有效的內部控制制度，以確保財務交易的合規性和準確性。這包括風險管理、審計和稽核等措施。

                5. 績效評估：財團法人應定期進行績效評估，以評估其業務目標的達成情況，並根據評估結果調整相應的策略和計劃。

                總的來說，數位發展部主管財團法人的預決算與會計處理及財務報告編製準則第二章將涵蓋財務管理的各個方面，以確保財團法人的財務運作合規且有效。詳細的內容可能需要參考具體的法規文件或相關指引。

--------------------------
{summaries}"""
messages = [
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template("{question}"),
]
CHAT_COMBINE_PROMPT = ChatPromptTemplate.from_messages(messages)


COMBINE_PROMPT_SELECTOR = ConditionalPromptSelector(
    default_prompt=COMBINE_PROMPT, conditionals=[(is_chat_model, CHAT_COMBINE_PROMPT)]
)
