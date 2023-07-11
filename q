[1mdiff --git a/python/lsst/summit/utils/efdUtils.py b/python/lsst/summit/utils/efdUtils.py[m
[1mindex e590e18..6882d8c 100644[m
[1m--- a/python/lsst/summit/utils/efdUtils.py[m
[1m+++ b/python/lsst/summit/utils/efdUtils.py[m
[36m@@ -235,11 +235,11 @@[m [mdef getEfdData(client, topic, *,[m
 [m
     nest_asyncio.apply()[m
     loop = asyncio.get_event_loop()[m
[31m-    ret = loop.run_until_complete(_getEfdData(client,[m
[31m-                                              topic,[m
[31m-                                              columns,[m
[31m-                                              begin.utc,[m
[31m-                                              end.utc))[m
[32m+[m[32m    ret = loop.run_until_complete(_getEfdData(client=client,[m
[32m+[m[32m                                              topic=topic,[m
[32m+[m[32m                                              begin=begin,[m
[32m+[m[32m                                              end=end,[m
[32m+[m[32m                                              columns=columns))[m
     if ret.empty and warn:[m
         log = logging.getLogger(__name__)[m
         log.warning(f"Topic {topic} is in the schema, but no data was returned by the query for the specified"[m
[36m@@ -247,21 +247,21 @@[m [mdef getEfdData(client, topic, *,[m
     return ret[m
 [m
 [m
[31m-async def _getEfdData(client, topic, columns, begin, end):[m
[32m+[m[32masync def _getEfdData(client, topic, begin, end, columns=None):[m
     """Get data for a topic from the EFD over the specified time range.[m
 [m
     Parameters[m
     ----------[m
     client : `lsst_efd_client.efd_helper.EfdClient`[m
[31m-        The EFD client to use.[m
[32m+[m[32m        The EFD client to use[m
     topic : `str`[m
         The topic to query.[m
[31m-    columns : `list` of `str`, optional[m
[31m-        The columns to query. If not specified, all columns are queried.[m
     begin : `astropy.Time`, optional[m
         The begin time for the query.[m
     end : `astropy.Time`, optional[m
         The end time for the query.[m
[32m+[m[32m    columns : `list` of `str`, optional[m
[32m+[m[32m        The columns to query. If not specified, all columns are returned.[m
 [m
     Returns[m
     -------[m
