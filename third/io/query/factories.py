import random
import time

from faker import Faker

from third.io.query.domain import (
    UserInfo,
    Gender,
    UserProfile,
    User,
)


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
            info=self.create_user_info(),
            profile=self.create_user_profile(),
        )

    def create_users(self, count: int) -> list[User]:
        return [self.create_user() for _ in range(count)]
