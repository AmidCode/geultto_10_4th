from sqlalchemy import select
from sqlalchemy.orm import (
    joinedload,
    contains_eager,
)

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
            UserInfo(name="sora", age=20, gender=Gender.MALE),
            UserProfile(nickname="aether", pic="/path/to/pic.png"),
        )
        user2 = User.create(
            UserInfo(name="hotaru", age=20, gender=Gender.FEMALE),
            UserProfile(nickname="lumine", pic="/path/to/pic.png"),
        )

        session.add_all([user1, user2])
        session.commit()

        #
        # -- contains_eager 사용시:
        #   SELECT user_info_tbl.id, ..., user_profile_tbl.id, ...
        #   FROM users_tbl
        #   JOIN user_info_tbl ON users_tbl.uid = user_info_tbl.uid
        #   JOIN user_profile_tbl ON users_tbl.uid = user_profile_tbl.uid
        #   WHERE user_info_tbl.name = 'hotaru'
        #
        stmt = (
            select(User)
            .join(UserInfo)  # INNER JOIN
            .where(UserInfo.name == "hotaru")
            .options(
                contains_eager(User.info),  # 이미 조인된 데이터 재사용
                contains_eager(User.profile),
            )
            .join(User.profile)  # profile 테이블 JOIN
        )

        #
        # -- joinedload 사용시:
        #   SELECT users_tbl.id, users_tbl.uid, ...
        #   FROM users_tbl
        #   JOIN user_info_tbl ON users_tbl.uid = user_info_tbl.uid
        #   JOIN user_profile_tbl ON users_tbl.uid = user_profile_tbl.uid
        #   LEFT OUTER JOIN user_info_tbl AS user_info_tbl_1 ON users_tbl.uid = user_info_tbl_1.uid  -- 추가 JOIN
        #   LEFT OUTER JOIN user_profile_tbl AS user_profile_tbl_1 ON users_tbl.uid = user_profile_tbl_1.uid  -- 추가 JOIN
        #   WHERE user_info_tbl.name = 'hotaru'
        #
        # stmt = (
        #     select(User)
        #     .join(UserInfo)  # INNER JOIN - info 테이블 JOIN
        #     .join(UserProfile)  # INNER JOIN - profile 테이블 JOIN
        #     .where(UserInfo.name == "hotaru")
        #     .options(
        #         joinedload(User.info),  # 이미 조인된 데이터 재사용
        #         joinedload(User.profile),
        #     )
        # )

        result = session.execute(stmt)
        user = result.scalar_one_or_none()

        assert user.info.name == "hotaru"
        assert user.profile.nickname == "lumine"

    def test_using_factory(self, session):
        factory = UserFactory()

        USERS = 1_000

        users = factory.create_users(USERS)
        assert len(users) == USERS
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

        print(len(queried))

        assert len(queried) == USERS
