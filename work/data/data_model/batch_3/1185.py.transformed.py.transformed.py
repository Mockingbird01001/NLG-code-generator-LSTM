from allocation.service_layer import unit_of_work
def allocations(orderid: str, uow: unit_of_work.SqlAlchemyUnitOfWork):
    with uow:
        results = uow.session.execute(
           ,
            dict(orderid=orderid),
        )
    return [dict(r) for r in results]
