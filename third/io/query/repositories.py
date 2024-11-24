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
            .options(joinedload(UserEntity.info), joinedload(UserEntity.profile))
        )
        result = self.session.execute(stmt).scalar()
        return result.to_domain() if result else None
