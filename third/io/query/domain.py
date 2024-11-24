import enum
import uuid
from dataclasses import dataclass


class Gender(enum.Enum):
    MALE = "male"
    FEMALE = "female"


@dataclass
class UserInfo:
    name: str
    age: int
    gender: Gender

    def __repr__(self) -> str:
        return (
            f"UserInfo(name='{self.name}', age={self.age}, gender={self.gender.value})"
        )


@dataclass
class UserProfile:
    nickname: str
    pic: str

    def __repr__(self) -> str:
        return f"UserProfile(nickname='{self.nickname}', pic='{self.pic}')"


class User:
    uid: uuid.UUID
    info: UserInfo
    profile: UserProfile

    def __init__(
        self,
        uid: uuid.UUID,
        info: UserInfo,
        profile: UserProfile,
    ):
        self.uid = uid
        self.info = info
        self.profile = profile

    def __repr__(self) -> str:
        return f"User(\n" f"  info={self.info},\n" f"  profile={self.profile}\n" f")"

    @classmethod
    def create(
        cls,
        info: UserInfo,
        profile: UserProfile,
    ):
        return cls(
            uid=uuid.uuid4(),
            info=info,
            profile=profile,
        )
