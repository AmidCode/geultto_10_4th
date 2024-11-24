# 도메인 객체
import random
import time
import uuid
from dataclasses import dataclass
from typing import Optional

from sqlalchemy import (
    select,
    between,
    func,
)
from sqlalchemy.orm import (
    joinedload,
    relationship,
)

from third.io.query.domain import Gender
from third.io.query.factories import UserFactory
from third.io.query.table import (
    mapper_registry,
    t_users,
    t_user_info,
    t_user_profile,
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
        self.min_id = 0
        self.max_id = 10_000

    def save(self, user: User) -> None:
        entity = UserEntity.from_domain(user)
        self.session.add(entity)
        self.session.commit()

    def bulk_save(self, users: list[User], batch_size: int = 1000) -> None:
        """
        대량의 사용자 데이터를 효율적으로 저장

        Args:
            users: 저장할 User 객체 리스트
            batch_size: 한 번에 처리할 배치 크기
        """
        for i in range(0, len(users), batch_size):
            batch = users[i : i + batch_size]
            self._execute_bulk_save(batch)

    def _execute_bulk_save(
        self,
        users: list[User],
    ) -> None:
        """
        실제 bulk insert를 실행하는 내부 메서드
        """
        try:
            # 1. Users 테이블 데이터 준비
            user_records = [{"uid": user.uid} for user in users]

            # 2. UserInfo 테이블 데이터 준비
            info_records = [
                {
                    "uid": user.uid,
                    "name": user.info.name,
                    "age": user.info.age,
                    "gender": user.info.gender,
                }
                for user in users
            ]

            # 3. UserProfile 테이블 데이터 준비
            profile_records = [
                {
                    "uid": user.uid,
                    "nickname": user.profile.nickname,
                    "pic": user.profile.pic,
                }
                for user in users
            ]

            # 4. Bulk insert 실행
            # Users 먼저 insert
            self.session.execute(t_users.insert(), user_records)
            self.session.execute(t_user_info.insert(), info_records)
            self.session.execute(t_user_profile.insert(), profile_records)

            # 5. 커밋
            self.session.commit()

        except Exception as e:
            self.session.rollback()
            raise e

    def save_all(self, users: list[User]) -> None:
        """
        이전 버전과의 호환성을 위한 메서드
        """
        return self.bulk_save(users)

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

    def _generate_random_range(self) -> tuple[int, int]:
        """임의의 범위를 생성하는 헬퍼 메서드"""
        start = random.randint(self.min_id, self.max_id - 1)
        # 시작점부터 최대 ID까지의 범위 중 임의의 끝점 선택
        end = random.randint(
            start, min(start + 10000, self.max_id)
        )  # 최대 1만개 범위로 제한
        return start, end

    def find_users_by_name_range(self) -> tuple[list[User], float, int]:
        """
        임의의 범위의 유저를 조회하고 실행 시간을 측정

        Returns:
            tuple[list[User], float, int]: (조회된 유저 목록, 실행 시간(초), 조회된 레코드 수)
        """
        start, end = self._generate_random_range()
        start_name = f"user_{start:06d}"
        end_name = f"user_{end:06d}"

        # 쿼리 실행 시간 측정 시작
        query_start_time = time.time()

        stmt = (
            select(UserEntity)
            .join(UserInfoEntity)
            .where(between(UserInfoEntity.name, start_name, end_name))
            .options(
                joinedload(UserEntity.info),
                joinedload(UserEntity.profile),
            )
        )

        results = self.session.execute(stmt).scalars().all()
        users = [result.to_domain() for result in results]

        # 실행 시간 계산
        execution_time = time.time() - query_start_time

        return users, execution_time, len(users)

    def run_random_range_query_test(self, num_tests: int = 10) -> list[dict]:
        """
        여러 번의 임의 범위 쿼리 테스트를 실행하고 결과를 반환

        Args:
            num_tests (int): 실행할 테스트 횟수

        Returns:
            List[dict]: 각 테스트의 결과 정보
        """
        test_results = []

        for i in range(num_tests):
            users, execution_time, record_count = self.find_users_by_name_range()

            test_result = {
                "test_number": i + 1,
                "execution_time": execution_time,
                "record_count": record_count,
                "records_per_second": (
                    record_count / execution_time if execution_time > 0 else 0
                ),
            }
            test_results.append(test_result)

            print(f"Test {i + 1}/{num_tests}:")
            print(f"  Records found: {record_count}")
            print(f"  Execution time: {execution_time:.4f} seconds")
            print(f"  Records/second: {test_result['records_per_second']:.2f}")
            print()

        return test_results

    def get_query_statistics(self) -> dict:
        """
        전체 유저 수와 이름 범위에 대한 통계 정보를 반환
        """
        total_users = self.session.query(func.count(UserEntity.id)).scalar()
        name_min = self.session.query(func.min(UserInfoEntity.name)).scalar()
        name_max = self.session.query(func.max(UserInfoEntity.name)).scalar()

        return {
            "total_users": total_users,
            "name_range": {"min": name_min, "max": name_max},
        }


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

    def test_random_user(self, session):
        repository = UserRepository(session)
        factory = UserFactory(seed=42)  # 재현 가능한 테스트를 위해 고정된 시드 사용

        # 대량의 테스트 데이터 생성
        users = factory.create_users(100_000)

        # 벌크 저장 수행 및 시간 측정
        start_time = time.time()
        repository.bulk_save(users)
        end_time = time.time()

        print(
            f"\nBulk insert time for 10,000 users: {end_time - start_time:.2f} seconds"
        )

        # 통계 정보 확인
        stats = repository.get_query_statistics()
        assert (
            stats["total_users"] >= 10_000
        )  # 이전 테스트의 데이터가 있을 수 있으므로 >=

        # 랜덤 범위 쿼리 테스트 실행 (5회)
        test_results = repository.run_random_range_query_test(num_tests=5)

        # 테스트 결과 검증
        assert len(test_results) == 5  # 5회 테스트 완료 확인

        for result in test_results:
            # 각 테스트 결과 검증
            assert result["execution_time"] > 0  # 실행 시간이 0보다 큼
            assert result["record_count"] >= 0  # 결과 건수가 0 이상
            assert result["records_per_second"] >= 0  # 초당 처리 건수가 0 이상

            # 성능 관련 기본적인 검증
            assert result["execution_time"] < 5.0  # 각 쿼리는 5초 이내 실행
            if result["record_count"] > 0:
                assert result["records_per_second"] > 100  # 초당 최소 100건 이상 처리

        # 특정 범위의 이름으로 조회 테스트
        first_user = repository.find_by_name("user_000001")
        assert first_user is not None
        assert first_user.info.name == "user_000001"

        # 중간 범위 유저 존재 확인
        middle_user = repository.find_by_name("user_005000")
        assert middle_user is not None
        assert middle_user.info.name == "user_005000"

        # 마지막 범위 유저 존재 확인
        last_user = repository.find_by_name("user_099999")
        assert last_user is not None
        assert last_user.info.name == "user_099999"

    def test_random_user_performance(self, session):
        """성능 테스트를 위한 별도의 테스트 메서드"""
        repository = UserRepository(session)
        factory = UserFactory(seed=42)

        batch_sizes = [10000]
        test_size = 100_000

        results = []

        for batch_size in batch_sizes:
            # 이전 테스트 데이터 정리
            self._clean_test_data(session)

            users = factory.create_users(test_size)

            start_time = time.time()
            repository.bulk_save(users, batch_size=batch_size)
            end_time = time.time()

            elapsed_time = end_time - start_time
            records_per_second = test_size / elapsed_time

            result = {
                "batch_size": batch_size,
                "total_time": elapsed_time,
                "records_per_second": records_per_second,
            }
            results.append(result)

            print(f"\nBatch size {batch_size}:")
            print(f"Total time: {elapsed_time:.2f} seconds")
            print(f"Records per second: {records_per_second:.2f}")

        # 최종 결과 출력
        print("\nPerformance Summary:")
        for result in results:
            print(f"Batch size {result['batch_size']}:")
            print(f"  - Time: {result['total_time']:.2f} seconds")
            print(f"  - Speed: {result['records_per_second']:.2f} records/second")

    def _clean_test_data(self, session):
        """테스트 데이터를 안전하게 정리하는 헬퍼 메서드"""
        try:
            # 외래 키 제약조건을 고려한 삭제 순서
            session.query(UserProfileEntity).delete()
            session.query(UserInfoEntity).delete()
            session.query(UserEntity).delete()
            session.commit()
        except Exception as e:
            session.rollback()
            raise e
