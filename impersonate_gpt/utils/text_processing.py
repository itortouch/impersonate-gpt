import pandas as pd
from whatstk import df_from_txt_whatsapp


def extract_firt_name(chat_df: pd.DataFrame) -> None:
    first_name_map = {name: name.split()[0] for name in chat_df.username.unique()}
    chat_df["first_name"] = chat_df["username"].map(first_name_map)


def filter_multimedia(chat_df: pd.DataFrame, language: str = "es") -> None:
    multimedia_texts = {"es": "<Multimedia omitido>"}
    text_mask = multimedia_texts.get(language, False)
    if text_mask:
        chat_df = chat_df.query("message != '<Multimedia omitido>'").reset_index(drop=True)
        return chat_df
    else:
        raise Exception(f"The language {language} is not supported yet.")


def replace_mentions_with_names(chat_df: pd.DataFrame, phone_name_mapper: dict) -> None:
    for phone, name in phone_name_mapper.items():
        chat_df["message"] = chat_df.message.str.replace(phone, name)
    return chat_df


def format_messages_as_chat_prompt(chat_df: pd.DataFrame) -> None:
    chat_df["prompt_format"] = chat_df["first_name"] + ": " + chat_df["message"]
    return chat_df


def build_completion_user_prompt_based(chat_df: pd.DataFrame, impersonated_user: str, n_context_messages: int = 10):
    imp_user_index = chat_df[chat_df["first_name"] == impersonated_user].index
    prompt_df = pd.DataFrame({"ideal_text_index": imp_user_index})
    prompt_df["promt_text_index"] = [list(range(i - n_context_messages, i)) for i in imp_user_index]
    prompt_df["completion"] = prompt_df["ideal_text_index"].map(chat_df["prompt_format"])
    prompt_df["prompt"] = prompt_df["promt_text_index"].map(lambda r: chat_df.loc[r, "prompt_format"].str.cat(sep="\n"))
    return prompt_df


def create_prompt_jsonl(
    whatsapp_chat_txt_path: str,
    phone_name_mapper: dict,
    impersonated_user: str,
    output_path: str = "./",
    language: str = "es",
    n_context_messages: int = 10,
):
    chat_df = df_from_txt_whatsapp(whatsapp_chat_txt_path)
    chat_df = filter_multimedia(chat_df, language)
    chat_df = replace_mentions_with_names(chat_df, phone_name_mapper)
    chat_df = format_messages_as_chat_prompt(chat_df)
    prompt_df = build_completion_user_prompt_based(chat_df, impersonated_user, n_context_messages)

    with open(
        f"{output_path}chat_prompts_{impersonated_user}_{language}_{n_context_messages}.jsonl", mode="bw"
    ) as writer:
        writer.write(prompt_df[["prompt", "completion"]].to_json(orient="records", lines=True).encode("utf8"))
