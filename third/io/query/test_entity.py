# 도메인 객체
import uuid
from dataclasses import dataclass
from typing import Optional

import pytest
from sqlalchemy import (
    select,
    create_engine,
)
from sqlalchemy.orm import (
    joinedload,
    sessionmaker,
    relationship,
)

from third.io.query.domain import Gender
from third.io.query.table import (
    mapper_registry,
    t_users,
    t_user_info,
    t_user_profile,
    metadata,
)


@dataclass
class UserInfo:
    name: str
    age: int
    gender: Gender


@dataclass
class UserProfile:
    nickname: str
    pic: str


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


# 영속성 객체
class UserInfoEntity:
    id: int
    uid: uuid.UUID
    name: str
    age: int
    gender: Gender

    @classmethod
    def from_domain(cls, user_info: UserInfo, uid: uuid.UUID) -> "UserInfoEntity":
        entity = cls()
        entity.uid = uid
        entity.name = user_info.name
        entity.age = user_info.age
        entity.gender = user_info.gender
        return entity

    def to_domain(self) -> UserInfo:
        return UserInfo(name=self.name, age=self.age, gender=self.gender)


class UserProfileEntity:
    id: int
    uid: uuid.UUID
    nickname: str
    pic: str

    @classmethod
    def from_domain(cls, profile: UserProfile, uid: uuid.UUID) -> "UserProfileEntity":
        entity = cls()
        entity.uid = uid
        entity.nickname = profile.nickname
        entity.pic = profile.pic
        return entity

    def to_domain(self) -> UserProfile:
        return UserProfile(nickname=self.nickname, pic=self.pic)


class UserEntity:
    id: int
    uid: uuid.UUID
    info: UserInfoEntity
    profile: UserProfileEntity

    @classmethod
    def from_domain(cls, user: User) -> "UserEntity":
        entity = cls()
        entity.uid = user.uid
        entity.info = UserInfoEntity.from_domain(user.info, user.uid)
        entity.profile = UserProfileEntity.from_domain(user.profile, user.uid)
        return entity

    def to_domain(self) -> User:
        return User(
            uid=self.uid, info=self.info.to_domain(), profile=self.profile.to_domain()
        )


# SQLAlchemy 매핑
mapper_registry.map_imperatively(
    UserEntity,
    t_users,
    properties={
        "info": relationship(
            UserInfoEntity,
            uselist=False,
            cascade="all, delete-orphan",
            overlaps="info",  # 추가
        ),
        "profile": relationship(
            UserProfileEntity,
            uselist=False,
            cascade="all, delete-orphan",
            overlaps="profile",  # 추가
        ),
    },
)

mapper_registry.map_imperatively(
    UserInfoEntity,
    t_user_info,
)

mapper_registry.map_imperatively(
    UserProfileEntity,
    t_user_profile,
)


# 리포지토리 계층
class UserRepository:
    def __init__(self, session):
        self.session = session

    def save(self, user: User) -> None:
        entity = UserEntity.from_domain(user)
        self.session.add(entity)
        self.session.commit()

    def find_by_name(self, name: str) -> Optional[User]:
        stmt = (
            select(UserEntity)
            .join(UserInfoEntity)
            .where(UserInfoEntity.name == name)
            .options(
                joinedload(UserEntity.info),
                joinedload(UserEntity.profile),
            )
        )
        result = self.session.execute(stmt).scalar()
        return result.to_domain() if result else None


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


# 테스트 코드
class TestUser:
    def test_user(self, session):
        repository = UserRepository(session)

        user1 = User.create(
            UserInfo(name="hotaru", age=20, gender=Gender.MALE),
            UserProfile(nickname="aether", pic="/path/to/pic.png"),
        )

        repository.save(user1)

        found_user = repository.find_by_name("hotaru")
        assert found_user.info.name == "hotaru"
        assert found_user.profile.nickname == "aether"
