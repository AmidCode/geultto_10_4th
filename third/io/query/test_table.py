from sqlalchemy import select
from sqlalchemy.orm import joinedload

from third.io.query.domain import (
    User,
    UserInfo,
    UserProfile,
    Gender,
)
from third.io.query.factories import UserFactory


class TestUser:
    def test_user(self, session):
        user1 = User.create(
            UserInfo(name="hotaru", age=20, gender=Gender.MALE),
            UserProfile(nickname="aether", pic="/path/to/pic.png"),
        )
        user2 = User.create(
            UserInfo(name="hikari", age=20, gender=Gender.FEMALE),
            UserProfile(nickname="lumine", pic="/path/to/pic.png"),
        )

        session.add_all([user1, user2])
        session.commit()

        stmt = (
            select(User)
            .join(UserInfo)
            .where(UserInfo.name == "hotaru")
            .options(joinedload(User.info), joinedload(User.profile))
        )
        result = session.execute(stmt)

        hotaru = result.scalar()

        assert hotaru.info.name == "hotaru"
        assert hotaru.profile.nickname == "aether"

    def test_using_factory(self, session):
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
