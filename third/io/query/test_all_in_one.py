import enum
import random
import time
import uuid
from dataclasses import dataclass

import pytest
from faker import Faker
from sqlalchemy import (
    Table,
    Column,
    Integer,
    String,
    ForeignKey,
    UUID,
    create_engine,
    select,
    Enum,
)
from sqlalchemy.orm import (
    registry,
    relationship,
    sessionmaker,
    joinedload,
)


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


# Registry와 MetaData 객체 생성
mapper_registry = registry()
metadata = mapper_registry.metadata

# 테이블 정의
t_users = Table(
    "users_tbl",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("uid", UUID, unique=True, nullable=False),
)

t_user_info = Table(
    "user_info_tbl",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("uid", UUID, ForeignKey("users_tbl.uid")),
    Column("name", String(50), unique=True, nullable=False),
    Column("age", Integer, nullable=False),
    Column("gender", Enum(Gender), nullable=False),
)

t_user_profile = Table(
    "user_profile_tbl",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("uid", UUID, ForeignKey("users_tbl.uid")),
    Column("nickname", String(50), nullable=False),
    Column("pic", String(256), nullable=True),
)


mapper_registry.map_imperatively(
    User,
    t_users,
    properties={
        "info": relationship(
            UserInfo,
            uselist=False,
            cascade="all, delete-orphan",
        ),
        "profile": relationship(
            UserProfile,
            uselist=False,
            cascade="all, delete-orphan",
        ),
    },
)

mapper_registry.map_imperatively(
    UserInfo,
    t_user_info,
)

mapper_registry.map_imperatively(
    UserProfile,
    t_user_profile,
)


@pytest.fixture(scope="function")
def session():
    engine = create_engine("sqlite:///:memory:")
    Session = sessionmaker(bind=engine)
    metadata.create_all(engine)
    session = Session()
    yield session
    session.close()
    metadata.drop_all(engine)
    engine.dispose()


class UserFactory:
    def __init__(self, seed: int = None):
        self.faker = Faker(["ko_KR"])  # 한국어 로케일 사용

        # seed가 None이면 현재 시각 기반 seed 사용
        if seed is None:
            # 방법 1: Unix timestamp (초 단위)
            seed = int(time.time())

            # 방법 2: Unix timestamp (밀리초 단위)
            # seed = int(time.time() * 1000)

            # 방법 3: 현재 시각의 특정 부분만 사용
            # now = datetime.now()
            # seed = int(f"{now.hour}{now.minute}{now.second}")

        print(f"Using seed: {seed}")  # 디버깅을 위해 사용된 seed 출력
        Faker.seed(seed)
        random.seed(seed)

    def create_user_info(self) -> UserInfo:
        return UserInfo(
            name=self.faker.name(),
            age=random.randint(20, 60),
            gender=random.choice([Gender.MALE, Gender.FEMALE]),
        )

    def create_user_profile(self) -> UserProfile:
        return UserProfile(
            nickname=self.faker.user_name(),  # 온라인 닉네임 스타일
            pic=f"/profile/images/{self.faker.uuid4()}.jpg",  # 가상의 이미지 경로
        )

    def create_user(self) -> User:
        return User.create(
            info=self.create_user_info(), profile=self.create_user_profile()
        )

    def create_users(self, count: int) -> list[User]:
        return [self.create_user() for _ in range(count)]


class TestUser:
    def test_user(self, session):
        factory = UserFactory()

        users = factory.create_users(5)
        assert len(users) == 5
        assert all(isinstance(u.info, UserInfo) for u in users)

        session.add_all(users)
        session.commit()

        stmt = (
            select(User)
            .join(UserInfo)
            .options(
                joinedload(User.info),
                joinedload(User.profile),
            )
        )
        result = session.execute(stmt)

        queried = result.scalars().all()

        for user in queried:
            print(user)
