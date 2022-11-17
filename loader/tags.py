import os
import numpy as np
import torch
import pandas as pd
import re
from collections import defaultdict
from sklearn.preprocessing import MultiLabelBinarizer

labels = {
    "INVALID": -1,
    "NONE": 0,
    "REQ_MINION_TARGET": 1,  # 随从目标
    "REQ_FRIENDLY_TARGET": 2,  # 友方目标
    "REQ_ENEMY_TARGET": 3,  # 敌方目标
    "REQ_DAMAGED_TARGET": 4,  # 损伤
    "REQ_MAX_SECRETS": 5,  # 最大奥秘
    "REQ_FROZEN_TARGET": 6,  # 冻结
    "REQ_CHARGE_TARGET": 7,  # 冲锋
    "REQ_TARGET_MAX_ATTACK": 8,  # 最大攻击力，有参数
    "REQ_NONSELF_TARGET": 9,  # 非自己
    "REQ_TARGET_WITH_RACE": 10,  # 种族 有参数
    "REQ_TARGET_TO_PLAY": 11,  # 小目标
    "REQ_NUM_MINION_SLOTS": 12,  # 随从数量插槽 有参数
    "REQ_WEAPON_EQUIPPED": 13,  # 武器装备，需要武器
    "REQ_ENOUGH_MANA": 14,
    "REQ_YOUR_TURN": 15,
    "REQ_NONSTEALTH_ENEMY_TARGET": 16,
    "REQ_HERO_TARGET": 17,  # 英雄
    "REQ_SECRET_ZONE_CAP": 18,
    "REQ_MINION_CAP_IF_TARGET_AVAILABLE": 19,
    "REQ_MINION_CAP": 20,
    "REQ_TARGET_ATTACKED_THIS_TURN": 21,
    "REQ_TARGET_IF_AVAILABLE": 22,  # 有目标如果用（抉择星辰降落，巫医）
    "REQ_MINIMUM_ENEMY_MINIONS": 23,  # 最少的地方随从，有参数
    "REQ_TARGET_FOR_COMBO": 24,  # 连击有目标
    "REQ_NOT_EXHAUSTED_ACTIVATE": 25,
    "REQ_UNIQUE_SECRET_OR_QUEST": 26,
    "REQ_TARGET_TAUNTER": 27,
    "REQ_CAN_BE_ATTACKED": 28,
    "REQ_ACTION_PWR_IS_MASTER_PWR": 29,
    "REQ_TARGET_MAGNET": 30,
    "REQ_ATTACK_GREATER_THAN_0": 31,
    "REQ_ATTACKER_NOT_FROZEN": 32,
    "REQ_HERO_OR_MINION_TARGET": 33,
    "REQ_CAN_BE_TARGETED_BY_SPELLS": 34,
    "REQ_SUBCARD_IS_PLAYABLE": 35,
    "REQ_TARGET_FOR_NO_COMBO": 36,
    "REQ_NOT_MINION_JUST_PLAYED": 37,
    "REQ_NOT_EXHAUSTED_HERO_POWER": 38,
    "REQ_CAN_BE_TARGETED_BY_OPPONENTS": 39,
    "REQ_ATTACKER_CAN_ATTACK": 40,
    "REQ_TARGET_MIN_ATTACK": 41,  # 有参数
    "REQ_CAN_BE_TARGETED_BY_HERO_POWERS": 42,
    "REQ_ENEMY_TARGET_NOT_IMMUNE": 43,
    "REQ_ENTIRE_ENTOURAGE_NOT_IN_PLAY": 44,
    "REQ_MINIMUM_TOTAL_MINIONS": 45,  # 需要最少随从数量，有参数
    "REQ_MUST_TARGET_TAUNTER": 46,  # 目标必须是嘲讽
    "REQ_UNDAMAGED_TARGET": 47,  # 目标未受伤
    "REQ_CAN_BE_TARGETED_BY_BATTLECRIES": 48,
    "REQ_STEADY_SHOT": 49,
    "REQ_MINION_OR_ENEMY_HERO": 50,
    "REQ_TARGET_IF_AVAILABLE_AND_DRAGON_IN_HAND": 51,  # 有龙牌在手
    "REQ_LEGENDARY_TARGET": 52,
    "REQ_FRIENDLY_MINION_DIED_THIS_TURN": 53,  # 需要一个死亡的友方随从在当前回合死亡
    "REQ_FRIENDLY_MINION_DIED_THIS_GAME": 54,  # 需要一个死亡的友方随从
    "REQ_ENEMY_WEAPON_EQUIPPED": 55,
    "REQ_TARGET_IF_AVAILABLE_AND_MINIMUM_FRIENDLY_MINIONS": 56,
    "REQ_TARGET_WITH_BATTLECRY": 57,
    "REQ_TARGET_WITH_DEATHRATTLE": 58,
    "REQ_TARGET_IF_AVAILABLE_AND_MINIMUM_FRIENDLY_SECRETS": 59,
    "REQ_SECRET_ZONE_CAP_FOR_NON_SECRET": 60,
    "REQ_TARGET_EXACT_COST": 61,
    "REQ_STEALTHED_TARGET": 62,
    "REQ_MINION_SLOT_OR_MANA_CRYSTAL_SLOT": 63,
    "REQ_MAX_QUESTS": 64,
    "REQ_TARGET_IF_AVAILABE_AND_ELEMENTAL_PLAYED_LAST_TURN": 65,
    "REQ_TARGET_NOT_VAMPIRE": 66,
    "REQ_TARGET_NOT_DAMAGEABLE_ONLY_BY_WEAPONS": 67,
    "REQ_NOT_DISABLED_HERO_POWER": 68,
    "REQ_MUST_PLAY_OTHER_CARD_FIRST": 69,
    "REQ_HAND_NOT_FULL": 70,
    "REQ_DRAG_TO_PLAY": 71,
    "REQ_TARGET_TO_PLAY2": 75,
}


def get_tag_name(id):
    try:
        index = list(labels.values()).index(id)
        return list(labels.keys())[index]
    except:
        pass
    return "INVALID"

def get_tag_id(name):
    if name in labels:
        return labels[name]
    return -1


class Tags:
    label_dim = 128
    def __init__(self, data_path):

        self.data_path = data_path
        self.df = None

    def construct_requirements(self, raw):
        splits = raw.split("~")
        dims = np.zeros((self.label_dim, ))
        for split in splits:
            if len(split) == 0:
                continue
            items = split.split(":")
            dims[int(items[0])] = 1 # int(items[1])
        return dims

    def load(self):
        data = defaultdict(list)
        ids = []
        descriptions = []
        requirements = []
        has_requirements = []
        with open(os.path.join(self.data_path, '2022-11-16-16-28-26 Tags.csv')) as f:
            for line in f.readlines():
                line = line.strip()
                indices = line.split("^")
                card_name = indices[0]
                card_id = indices[1]
                card_description = indices[2]
                for identifier in re.findall("<[^>]*>", card_description):
                    card_description = card_description.replace(identifier, "")
                card_description = card_description.replace("[x]", "").replace("@", "0").replace("$", "")
                card_requirements_raw = indices[3]
                card_tags_raw = indices[4]
                card_requirements = self.construct_requirements(card_requirements_raw)
                if len(card_requirements_raw) > 0:
                    has_requirement = True
                else:
                    has_requirement = False
                ids.append(card_id)
                descriptions.append(card_description)
                requirements.append(card_requirements)
                has_requirements.append(has_requirement)


        self.df = pd.DataFrame(list(zip(ids,
                                   descriptions,
                                   requirements,
                                   has_requirements
                                   )), columns=['id', 'text', 'requirements', 'has_requirement'])

    def load_test(self):
        pass
